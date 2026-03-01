import numpy as np

import elastica as ea
import elastica_rigid as er
import elastica_plugins as ep  # Local plugins

from config import FlockingConfig

from tqdm import tqdm


class Simulator(
    ea.BaseSystemCollection,
    ea.Forcing,
    ea.CallBacks,
    ep.Flocking,
):
    pass


def create_robot(config: FlockingConfig, rng: np.random.Generator) -> er.SE2RigidBody:
    """
    Return random robot according to the config
    """
    lx, ly = config.box_size
    position = rng.uniform(0.0, 1.0, size=(2,))
    position[0] *= lx
    position[1] *= lx

    theta = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

    return er.SE2RigidBody.create_body(
        initial_position=position,
        initial_direction=direction,
        mass=float(config.mass),
        inertia=float(config.inertia),
    )


config: FlockingConfig = FlockingConfig()
rng: np.random.Generator = np.random.default_rng()

simulator = Simulator()
simulator.enable_block_supports(er.SE2RigidBody, er.MemoryBlockSE2Body)

for _ in tqdm(range(config.n_bodies), desc="init-robot"):
    robot = create_robot(config, rng)
    simulator.append(robot)


simulator.configure_flocking(
    flocking_block=er.MemoryBlockSE2Body,
    box_size=config.box_size,
).using(
    ep.VicsekModel,
    box_size=config.box_size,
    config=config,
    rng=rng,
)

simulator.finalize()

stepper = ea.PositionVerlet()
time = 0
for _ in tqdm(range(10), desc="simulation"):
    time = stepper.step(simulator, time, config.dt)
