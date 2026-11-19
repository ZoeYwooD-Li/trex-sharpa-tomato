"""Standalone packaging and import boundaries, independent of the old checkout."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from lycheesim import cli
from lycheesim.core.demo_registry import list_demos, repository_root

ROOT = Path(__file__).resolve().parents[1]


def test_discovery_cannot_follow_another_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("LYCHEESIM_PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert repository_root() == ROOT
    assert [demo.name for demo in list_demos()] == ["cherry-tomato-pick"]


def test_current_default_keeps_recording_and_collection():
    assert cli.DEFAULT_CONFIG == ROOT / "configs/cherry_tomato_pick_sharpa.yaml"
    cfg = yaml.safe_load(cli.DEFAULT_CONFIG.read_text())
    assert cfg["recording"]["enabled"] is True
    assert cfg["recording"]["auto_increment"] is True
    assert cfg["recording"]["randomization"]["enabled"] is True
    for key in ("carrier_mjcf", "sharpa_left_mjcf", "sharpa_right_mjcf", "fr3_visual_mesh_dir"):
        assert (ROOT / cfg["robot"][key]).exists()


@pytest.mark.parametrize("command", [[], ["run"], ["batch"], ["quality"]])
def test_cli_help(command):
    result = subprocess.run([sys.executable, "-m", "lycheesim.cli", *command, "--help"],
                            cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_no_imports_of_unrelated_demos_or_symlinked_assets():
    for path in (ROOT / "demos/cherry_tomato_pick").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            assert all(not name.startswith("demos.") or
                       name.startswith("demos.cherry_tomato_pick") for name in names), path
    for asset_dir in (ROOT / "demos/assets", ROOT / "demos/cherry_tomato_pick/assets"):
        assert not any(path.is_symlink() for path in asset_dir.rglob("*"))


def test_snapshot_records_uncommitted_source_and_helper_hashes():
    manifest = json.loads((ROOT / "provenance/source_snapshot.json").read_text())
    assert len(manifest["source_commit"]) == 40
    assert "demos/cherry_tomato_pick/batch.py" in manifest["copied_source_sha256"]
    assert manifest["source_working_tree_status"]
    assert len(manifest["extracted_helper_source_sha256"]) == 3
