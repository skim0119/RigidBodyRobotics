from dataclasses import dataclass


@dataclass(frozen=True)
class UiConfig:
    window_width: int = 1280
    window_height: int = 600
    left_panel_width: int = 760


DEFAULT_UI_CONFIG = UiConfig()
