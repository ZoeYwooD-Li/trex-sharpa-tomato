"""Run the scripted Newton cherry-tomato harvesting demo."""

from demos.cherry_tomato_pick.scene import build
from lycheesim.engine.runner import main


if __name__ == "__main__":
    main(build, demo_name="cherry_tomato_pick")
