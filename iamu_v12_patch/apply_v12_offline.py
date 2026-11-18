#!/usr/bin/env python3
"""Offline, Git-free installer for the reviewed IAMU V1.2 patch.

Usage:
  python apply_v12_offline.py --repo /home/moco/code/iamu/iamu_v12 --check
  python apply_v12_offline.py --repo /home/moco/code/iamu/iamu_v12 --apply

The script MUST sit next to apply_v12.py and payload/ extracted from the
previous bundle. It does not read or write .pt checkpoint files, nor results/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from zipfile import ZIP_DEFLATED, ZipFile

# Exact Git blob IDs of six source files at audited repository commit
# 2d96981a38434014f14e31be5641d422d28835f8.
EXPECTED_BASE_BLOBS = {
    "iamu_dynamic/rl_env_v11.py": "08bd876da3ae0fb6f0753a2169a7d7d541576050",
    "iamu_dynamic/rl_ddqn_v11.py": "990a795cde401dfc136ef70d9e070e4b0b3ca173",
    "v11_protocol/v11_actors.py": "53105787a4936132b43975aa8b08a3b927b4bd5e",
    "v11_protocol/indexed_replay.py": "065a06026d8fbe29185d3a761ab03edc9316ed47",
    "v11_protocol/fixed_label_probe.py": "5b264f2a88a9c07c3897c554ed5987bce963e8ec",
    "v11_protocol/train_v11.py": "50338334289ef7cc8c6f7583eba419f295eb80ea",
}
NEEDED = (
    "iamu_dynamic/__init__.py",
    "iamu_dynamic/core.py",
    "iamu_dynamic/risk_selection.py",
    "iamu_dynamic/rl_experiment_v11.py",
    "parallel_recovery_v081.py",
    "v11_protocol/__init__.py",
    "v11_protocol/rounds.py",
    "v11_protocol/validation_v11.py",
    "v11_protocol/launch_four_gpu.py",
    "tests_v11/test_indexed_and_protocol.py",
    "tests_v11/test_real_env_integration.py",
)


def blob_id(blob: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(blob)).encode("ascii") + b"\0" + blob).hexdigest()


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def load_patcher(base: Path):
    filename = base / "apply_v12.py"
    if not filename.is_file():
        raise FileNotFoundError(f"Missing original patcher beside offline installer: {filename}")
    spec = importlib.util.spec_from_file_location("iamu_v12_legacy_patcher", filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import patch code: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_changes(base: Path, root: Path):
    mod = root / "iamu_d3dqn"
    cases = root / "2024LiERLDDE" / "case_code"
    for rel in NEEDED:
        if not (mod / rel).is_file():
            raise FileNotFoundError(f"Required V1.1 code missing: {mod / rel}")
    for case_id in range(1, 10):
        if not (cases / f"{case_id}.txt").is_file():
            raise FileNotFoundError(f"Training/validation case missing: {cases / f'{case_id}.txt'}")
    if not cases.is_dir():
        raise FileNotFoundError(f"Dataset directory missing: {cases}")
    patcher = load_patcher(base)
    if set(patcher.PATCHES) != set(EXPECTED_BASE_BLOBS):
        raise ValueError("Unexpected version of apply_v12.py: PATCHES differ from audited list")
    existing: dict[Path, bytes] = {}
    changes: dict[Path, bytes] = {}
    for rel, transform in patcher.PATCHES.items():
        path = mod / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        actual = blob_id(raw)
        if actual != EXPECTED_BASE_BLOBS[rel]:
            raise ValueError(
                f"Source mismatch (or patch already applied): {path}\n"
                f"  observed Git blob: {actual}\n"
                f"  expected Git blob: {EXPECTED_BASE_BLOBS[rel]}\n"
                "No files changed. Compare source before attempting an override."
            )
        modified = transform(raw.decode("utf-8"))
        if modified == raw.decode("utf-8"):
            raise AssertionError(f"Patch had no effect: {path}")
        ast.parse(modified, filename=str(path))
        existing[path] = raw
        changes[path] = modified.encode("utf-8")

    payload_dir = base / "payload"
    if not payload_dir.is_dir():
        raise FileNotFoundError(f"Missing extracted patch payload: {payload_dir}")
    payloads = [p for p in sorted(payload_dir.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    if len(payloads) < 15:
        raise ValueError(f"Unexpectedly incomplete V1.2 payload: {len(payloads)} files")
    for source in payloads:
        target = mod / source.relative_to(payload_dir)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing overwrite (may already be installed): {target}")
        raw = source.read_bytes()
        if target.suffix == ".py":
            ast.parse(raw.decode("utf-8"), filename=str(target))
        changes[target] = raw
    return existing, changes, len(payloads)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Root holding iamu_d3dqn and 2024LiERLDDE")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Read-only preflight")
    mode.add_argument("--apply", action="store_true", help="Backup and install V1.2")
    a = p.parse_args()
    root = Path(a.repo).expanduser().resolve()
    base = Path(__file__).resolve().parent
    if not root.is_dir():
        p.error(f"Research root does not exist: {root}")
    existing, changes, payload_count = collect_changes(base, root)
    print(f"[V1.2 offline] preflight passed: {len(existing)} modified files, "
          f"{payload_count} new files, 9 case files; root={root}")
    print("[V1.2 offline] no Git access required; no .pt or results/ paths will be touched")
    if a.check:
        return

    backups = root.parent / "v12_patch_backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backups / f"pre_v12_sources_{stamp}_{os.getpid()}.zip"
    manifest = {
        "original_source_commit": "2d96981a38434014f14e31be5641d422d28835f8",
        "root": str(root),
        "modified_sources": {},
        "created_files": [],
        "timestamp_utc": stamp,
    }
    with ZipFile(backup, "x", compression=ZIP_DEFLATED) as zf:
        for path, raw in existing.items():
            rel = path.relative_to(root).as_posix()
            zf.writestr(rel, raw)
            manifest["modified_sources"][rel] = {"sha256": sha256(raw), "git_blob": blob_id(raw)}
        manifest["created_files"] = [p.relative_to(root).as_posix() for p in changes if p not in existing]
        zf.writestr("__v12_backup_manifest__.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[V1.2 offline] original source files backed up: {backup}")
    written = []
    try:
        for path, raw in changes.items():
            if path not in existing and (path.exists() or path.is_symlink()):
                raise FileExistsError(f"Concurrent creation detected: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + f".v12_{os.getpid()}.tmp")
            try:
                with open(temp, "xb") as f:
                    f.write(raw)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp, path)
            finally:
                if temp.exists():
                    temp.unlink()
            written.append(path)
    except Exception:
        for path in reversed(written):
            if path in existing:
                path.write_bytes(existing[path])
            else:
                path.unlink(missing_ok=True)
        raise
    marker = root / "iamu_d3dqn" / "V12_OFFLINE_INSTALL.json"
    if not marker.exists():
        marker.write_text(json.dumps({"base_commit": manifest["original_source_commit"],
                                      "backup": str(backup),
                                      "installed_files": manifest["created_files"]},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[V1.2 offline] applied successfully. Original checkpoint folders untouched.\n"
          f"  Source backup: {backup}\n  Marker: {marker}")
    print("Next: cd <repo>/iamu_d3dqn && python -m unittest discover -s tests_v11 -p 'test_*.py' -v")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, SyntaxError, AssertionError) as exc:
        print(f"[V1.2 offline] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
