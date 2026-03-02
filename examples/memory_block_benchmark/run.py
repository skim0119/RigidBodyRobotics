from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, replace
from typing import Iterable

import click
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import elastica as ea
import elastica_rigid as er


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
):
    pass


@dataclass
class BenchmarkConfig:
    n_bodies: int = 1000
    box_size: tuple[float, float] = (20.0, 20.0)
    dt: float = 0.01
    mass: float = 1.0
    inertia: float = 0.2
    initial_speed: float = 1.0
    initial_omega_scale: float = 0.5


def create_robot(config: BenchmarkConfig, rng: np.random.Generator) -> er.SE2RigidBody:
    lx, ly = config.box_size
    position = rng.uniform(0.0, 1.0, size=(2,))
    position[0] *= lx
    position[1] *= ly

    theta = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    speed = config.initial_speed * rng.uniform(0.8, 1.2)
    velocity = speed * direction
    omega = rng.uniform(-config.initial_omega_scale, config.initial_omega_scale)

    return er.SE2RigidBody(
        position=position,
        direction=direction,
        mass=float(config.mass),
        inertia=float(config.inertia),
        initial_velocity=velocity,
        initial_omega=omega,
    )


def build_simulator(
    config: BenchmarkConfig,
    rng: np.random.Generator,
    memory_block_cls: type[er.MemoryBlockSE2Body],
) -> Simulator:
    simulator = Simulator()
    simulator.enable_block_supports(er.SE2RigidBody, memory_block_cls)

    for _ in range(config.n_bodies):
        simulator.append(create_robot(config, rng))

    simulator.finalize()
    return simulator


def run_steps(simulator: Simulator, dt: float, steps: int) -> float:
    stepper = ea.PositionVerlet()
    t = 0.0
    for _ in range(steps):
        t = stepper.step(simulator, t, dt)
    return t


def time_case(
    config: BenchmarkConfig,
    memory_block_cls: type[er.MemoryBlockSE2Body],
    *,
    steps: int,
    warmup_steps: int,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    elapsed: list[float] = []

    for k in range(repeats):
        rng = np.random.default_rng(seed + k)
        simulator = build_simulator(config, rng, memory_block_cls)

        if warmup_steps > 0:
            run_steps(simulator, config.dt, warmup_steps)

        t0 = time.perf_counter()
        run_steps(simulator, config.dt, steps)
        elapsed.append(time.perf_counter() - t0)

    mean_wall = statistics.fmean(elapsed)
    std_wall = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    mean_step = mean_wall / steps
    return mean_wall, std_wall, mean_step


def benchmark_memory_structure(
    base_config: BenchmarkConfig,
    *,
    n_bodies: int,
    steps: int,
    warmup_steps: int,
    repeats: int,
    seed: int,
    segment_sizes: Iterable[int],
) -> list[dict[str, float | int | str]]:
    cfg = replace(base_config, n_bodies=int(n_bodies))
    results: list[dict[str, float | int | str]] = []

    cases: list[tuple[str, type[er.MemoryBlockSE2Body]]] = []
    for layout in ("soa", "aos"):
        cases.append(
            (
                f"{layout}/single",
                er.MemoryBlockSE2Body.configure(
                    storage_layout=layout,
                    blocking_policy="single",
                ),
            )
        )
        for seg in segment_sizes:
            cases.append(
                (
                    f"{layout}/segmented/{seg}",
                    er.MemoryBlockSE2Body.configure(
                        storage_layout=layout,
                        blocking_policy="segmented",
                        segment_size=int(seg),
                    ),
                )
            )

    for label, block_cls in cases:
        mean_wall, std_wall, mean_step = time_case(
            cfg,
            block_cls,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            seed=seed,
        )
        results.append(
            {
                "case": label,
                "n_bodies": cfg.n_bodies,
                "steps": steps,
                "mean_wall_s": mean_wall,
                "std_wall_s": std_wall,
                "mean_step_s": mean_step,
                "steps_per_s": steps / mean_wall,
            }
        )
        print(
            "[memory] "
            f"case={label} bodies={cfg.n_bodies} "
            f"mean_step_ms={1000.0 * mean_step:.6f} "
            f"steps_per_s={steps / mean_wall:.2f} "
            f"wall={mean_wall:.4f}±{std_wall:.4f}s"
        )

    results.sort(key=lambda r: float(r["mean_step_s"]))
    return results


def benchmark_default_scaling(
    base_config: BenchmarkConfig,
    *,
    body_counts: Iterable[int],
    steps: int,
    warmup_steps: int,
    repeats: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    cases = [
        (
            "soa",
            er.MemoryBlockSE2Body.configure(
                storage_layout="soa",
                blocking_policy="single",
            ),
        ),
        (
            "aos",
            er.MemoryBlockSE2Body.configure(
                storage_layout="aos",
                blocking_policy="single",
            ),
        ),
    ]

    for layout, block_cls in cases:
        for n_bodies in body_counts:
            cfg = replace(base_config, n_bodies=int(n_bodies))
            mean_wall, std_wall, mean_step = time_case(
                cfg,
                block_cls,
                steps=steps,
                warmup_steps=warmup_steps,
                repeats=repeats,
                seed=seed,
            )
            results.append(
                {
                    "layout": layout,
                    "n_bodies": cfg.n_bodies,
                    "steps": steps,
                    "mean_wall_s": mean_wall,
                    "std_wall_s": std_wall,
                    "mean_step_s": mean_step,
                    "steps_per_s": steps / mean_wall,
                }
            )
            print(
                f"[scaling/{layout}] "
                f"bodies={cfg.n_bodies} "
                f"mean_step_ms={1000.0 * mean_step:.6f} "
                f"steps_per_s={steps / mean_wall:.2f} "
                f"wall={mean_wall:.4f}±{std_wall:.4f}s"
            )

    results.sort(key=lambda r: (str(r["layout"]), int(r["n_bodies"])))
    return results


def _print_table_memory(results: list[dict[str, float | int | str]]) -> None:
    print("\n=== Scenario 1: Memory Structure Sweep ===")
    print(
        "rank | case               | bodies | mean_step_ms | steps/s | mean_wall_s ± std"
    )
    for rank, r in enumerate(results, start=1):
        print(
            f"{rank:>4d} | "
            f"{str(r['case']):<18s} | "
            f"{int(r['n_bodies']):>6d} | "
            f"{1000.0 * float(r['mean_step_s']):>12.6f} | "
            f"{float(r['steps_per_s']):>7.2f} | "
            f"{float(r['mean_wall_s']):.4f} ± {float(r['std_wall_s']):.4f}"
        )


def _print_table_scaling(results: list[dict[str, float | int | str]]) -> None:
    print("\n=== Scenario 2: SOA vs AOS Scaling ===")
    print("layout | bodies | mean_step_ms | steps/s | mean_wall_s ± std")
    for r in results:
        print(
            f"{str(r['layout']):>6s} | "
            f"{int(r['n_bodies']):>6d} | "
            f"{1000.0 * float(r['mean_step_s']):>12.6f} | "
            f"{float(r['steps_per_s']):>7.2f} | "
            f"{float(r['mean_wall_s']):.4f} ± {float(r['std_wall_s']):.4f}"
        )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-16:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _compute_scaling_fit_report(
    results: list[dict[str, float | int | str]],
) -> dict[str, dict[str, float | str]]:
    report: dict[str, dict[str, float | str]] = {}
    for layout in ("soa", "aos"):
        rows = [r for r in results if r["layout"] == layout]
        rows.sort(key=lambda r: int(r["n_bodies"]))
        if len(rows) < 2:
            continue

        n = np.asarray([int(r["n_bodies"]) for r in rows], dtype=np.float64)
        t_ms = np.asarray(
            [1000.0 * float(r["mean_step_s"]) for r in rows], dtype=np.float64
        )

        # Linear model: T = a*N + b
        lin_a, lin_b = np.polyfit(n, t_ms, 1)
        t_lin = lin_a * n + lin_b
        r2_lin = _r2_score(t_ms, t_lin)

        # Quadratic model: T = a*N^2 + b*N + c
        if len(rows) >= 3:
            q_a, q_b, q_c = np.polyfit(n, t_ms, 2)
            t_quad = q_a * n * n + q_b * n + q_c
            r2_quad = _r2_score(t_ms, t_quad)
        else:
            q_a = q_b = q_c = float("nan")
            r2_quad = float("nan")

        # Power-law model: T = k * N^p
        p, logk = np.polyfit(np.log(n), np.log(t_ms), 1)
        k = float(np.exp(logk))
        t_pow = k * np.power(n, p)
        r2_pow = _r2_score(t_ms, t_pow)

        if p < 1.15:
            verdict = "near-linear"
        elif p < 1.7:
            verdict = "superlinear"
        else:
            verdict = "near-quadratic"

        report[layout] = {
            "k": k,
            "p": float(p),
            "r2_pow": float(r2_pow),
            "lin_a": float(lin_a),
            "lin_b": float(lin_b),
            "r2_lin": float(r2_lin),
            "q_a": float(q_a) if np.isfinite(q_a) else float("nan"),
            "q_b": float(q_b) if np.isfinite(q_b) else float("nan"),
            "q_c": float(q_c) if np.isfinite(q_c) else float("nan"),
            "r2_quad": float(r2_quad) if np.isfinite(r2_quad) else float("nan"),
            "verdict": verdict,
        }
    return report


def _print_scaling_fit_report(results: list[dict[str, float | int | str]]) -> None:
    print("\n=== Scaling Fit Report (per layout) ===")
    report = _compute_scaling_fit_report(results)
    for layout in ("soa", "aos"):
        fit = report.get(layout)
        if fit is None:
            continue
        print(
            f"[{layout.upper()}] power: T={float(fit['k']):.4e}*N^{float(fit['p']):.4f} "
            f"(R^2={float(fit['r2_pow']):.5f}) -> {str(fit['verdict'])}"
        )
        print(
            f"[{layout.upper()}] linear: T={float(fit['lin_a']):.4e}*N + {float(fit['lin_b']):.4e} "
            f"(R^2={float(fit['r2_lin']):.5f})"
        )
        if np.isfinite(float(fit["r2_quad"])):
            print(
                f"[{layout.upper()}] quad:   T={float(fit['q_a']):.4e}*N^2 + "
                f"{float(fit['q_b']):.4e}*N + {float(fit['q_c']):.4e} "
                f"(R^2={float(fit['r2_quad']):.5f})"
            )


def _save_memory_plot(
    results: list[dict[str, float | int | str]],
    output_path: str,
) -> None:
    labels = [str(r["case"]) for r in results]
    mean_ms = [1000.0 * float(r["mean_step_s"]) for r in results]
    std_ms = [1000.0 * float(r["std_wall_s"]) / int(r["steps"]) for r in results]

    fig_h = max(4.0, 0.36 * len(labels))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, mean_ms, xerr=std_ms, alpha=0.9, color="#4c78a8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Step Time (ms)")
    ax.set_title("Memory Structure Benchmark (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _save_scaling_plot(
    results: list[dict[str, float | int | str]], output_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"soa": "#1f77b4", "aos": "#d62728"}
    report = _compute_scaling_fit_report(results)

    for layout in ("soa", "aos"):
        rows = [r for r in results if r["layout"] == layout]
        rows.sort(key=lambda r: int(r["n_bodies"]))
        if not rows:
            continue

        n = np.asarray([int(r["n_bodies"]) for r in rows], dtype=np.float64)
        mean_ms = np.asarray(
            [1000.0 * float(r["mean_step_s"]) for r in rows], dtype=np.float64
        )
        std_ms = np.asarray(
            [1000.0 * float(r["std_wall_s"]) / int(r["steps"]) for r in rows],
            dtype=np.float64,
        )
        color = colors[layout]

        ax.errorbar(
            n,
            mean_ms,
            yerr=std_ms,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=color,
            label=f"{layout.upper()} data",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Bodies (log scale)")
    ax.set_ylabel("Mean Step Time (ms)")
    ax.set_title("SOA vs AOS Scaling")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    lines: list[str] = []
    for layout in ("soa", "aos"):
        fit = report.get(layout)
        if fit is None:
            continue
        lines.append(
            f"{layout.upper()}: p={float(fit['p']):.3f}, "
            f"R2_pow={float(fit['r2_pow']):.4f}, {str(fit['verdict'])}"
        )
    if lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--steps", type=int, default=300, show_default=True, help="Timed steps per repeat."
)
@click.option(
    "--warmup-steps",
    type=int,
    default=30,
    show_default=True,
    help="Warmup steps before timing (per repeat).",
)
@click.option(
    "--repeats", type=int, default=3, show_default=True, help="Repeat count per case."
)
@click.option(
    "--seed", type=int, default=42, show_default=True, help="Base random seed."
)
@click.option(
    "--structure-bodies",
    type=int,
    default=100_000,
    show_default=True,
    help="Body count for memory-structure sweep.",
)
@click.option(
    "--segment-sizes",
    type=int,
    multiple=True,
    default=(64, 128, 256, 512, 1024),
    show_default=True,
    help="Segment sizes for segmented-block cases. Repeat option to pass multiple values.",
)
@click.option(
    "--scaling-bodies",
    type=int,
    multiple=True,
    default=(100, 200, 500, 1000, 2000, 5000),
    show_default=True,
    help="Body counts for default scaling sweep. Repeat option to pass multiple values.",
)
@click.option(
    "--memory-plot",
    type=str,
    default="benchmark_memory_structure.png",
    show_default=True,
    help="Output PNG path for memory-structure plot.",
)
@click.option(
    "--scaling-plot",
    type=str,
    default="benchmark_default_scaling.png",
    show_default=True,
    help="Output PNG path for default-scaling plot.",
)
def main(
    steps: int,
    warmup_steps: int,
    repeats: int,
    seed: int,
    structure_bodies: int,
    segment_sizes: tuple[int, ...],
    scaling_bodies: tuple[int, ...],
    memory_plot: str,
    scaling_plot: str,
) -> None:
    base_cfg = BenchmarkConfig()

    memory_results = benchmark_memory_structure(
        base_cfg,
        n_bodies=structure_bodies,
        steps=steps,
        warmup_steps=warmup_steps,
        repeats=repeats,
        seed=seed,
        segment_sizes=segment_sizes,
    )
    _print_table_memory(memory_results)
    _save_memory_plot(memory_results, memory_plot)

    scaling_results = benchmark_default_scaling(
        base_cfg,
        body_counts=scaling_bodies,
        steps=steps,
        warmup_steps=warmup_steps,
        repeats=repeats,
        seed=seed,
    )
    _print_table_scaling(scaling_results)
    _print_scaling_fit_report(scaling_results)
    _save_scaling_plot(scaling_results, scaling_plot)


if __name__ == "__main__":
    main()
