"""Task-specific physical interfaces and assets for tomato harvesting."""

from .assets import (
    cherry_tomato_all_fruit_visual_assets,
    cherry_tomato_robot_asset,
    cherry_tomato_visual_assets,
)
from .attachment import (
    CherryTomatoAttachmentFacade,
    CherryTomatoAttachmentViews,
)

__all__ = [
    "CherryTomatoAttachmentFacade",
    "CherryTomatoAttachmentViews",
    "cherry_tomato_all_fruit_visual_assets",
    "cherry_tomato_robot_asset",
    "cherry_tomato_visual_assets",
]
