"""Convert ASCII STL exports in rewind_glove_assembly/meshes to binary STL for MuJoCo."""

from __future__ import annotations

import sys
from pathlib import Path

import trimesh

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import config as cfg  # noqa: E402


def main() -> None:
    cfg.MESH_DIR.mkdir(parents=True, exist_ok=True)
    names = {Path(name).name for name in cfg.LINK_TO_STL.values()}
    missing: list[str] = []
    for name in sorted(names):
        src = cfg.MESH_DIR_SRC / name
        if not src.is_file():
            missing.append(name)
            continue
        mesh = trimesh.load(str(src), force="mesh")
        mesh.export(str(cfg.MESH_DIR / name), file_type="stl")
        print(f"  {name}")
    if missing:
        print("Missing source STL(s):", ", ".join(missing))
        sys.exit(1)
    print(f"Wrote {len(names) - len(missing)} file(s) to {cfg.MESH_DIR}")


if __name__ == "__main__":
    main()
