from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "models"
DEFAULT_DEST_DIR = ROOT / "models" / "production"


def resolve_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def find_run_tag(source_dir: Path, requested_tag: str | None) -> str:
    if requested_tag:
        manifest = source_dir / f"run_manifest_{requested_tag}.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Requested run tag not found: {requested_tag}")
        return requested_tag

    manifests = sorted(source_dir.glob("run_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No run manifests found in {source_dir}")

    latest = manifests[-1]
    return latest.stem.replace("run_manifest_", "")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def copy_artifact(path_value: str, dest_dir: Path) -> str:
    source_path = resolve_path(path_value, ROOT)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing artifact: {source_path}")

    target_path = dest_dir / source_path.name
    shutil.copy2(source_path, target_path)
    return target_path.relative_to(ROOT).as_posix()


def prepare_bundle(run_tag: str, source_dir: Path, dest_dir: Path, clean_dest: bool) -> list[str]:
    manifest_path = source_dir / f"run_manifest_{run_tag}.json"
    manifest = read_json(manifest_path)

    if clean_dest and dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []

    manifest["feature_schema_path"] = copy_artifact(manifest["feature_schema_path"], dest_dir)
    copied_files.append(manifest["feature_schema_path"])

    for model_key in ["electricity", "heat"]:
        model_path = manifest["models"][model_key]
        manifest["models"][model_key] = copy_artifact(model_path, dest_dir)
        copied_files.append(manifest["models"][model_key])

    artifact_keys = [
        "runtime_model",
        "electricity_imputer",
        "heat_imputer",
        "pipeline_meta",
        "slice_calibrators",
    ]

    for key in artifact_keys:
        value = manifest["pipeline_artifacts"].get(key)
        if not value:
            continue
        manifest["pipeline_artifacts"][key] = copy_artifact(value, dest_dir)
        copied_files.append(manifest["pipeline_artifacts"][key])

    output_manifest = dest_dir / f"run_manifest_{run_tag}.json"
    write_json(output_manifest, manifest)
    copied_files.append(output_manifest.relative_to(ROOT).as_posix())

    latest_path = dest_dir / "LATEST_RUN_TAG.txt"
    latest_path.write_text(run_tag, encoding="utf-8")
    copied_files.append(latest_path.relative_to(ROOT).as_posix())

    return copied_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare minimal production model bundle for deployment.")
    parser.add_argument("--run-tag", dest="run_tag", default=None, help="Run tag to package. Defaults to latest manifest.")
    parser.add_argument("--source-dir", dest="source_dir", default=str(DEFAULT_SOURCE_DIR), help="Directory containing training manifests/artifacts.")
    parser.add_argument("--dest-dir", dest="dest_dir", default=str(DEFAULT_DEST_DIR), help="Destination directory for deployment bundle.")
    parser.add_argument("--no-clean", dest="no_clean", action="store_true", help="Do not remove destination directory before packaging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir

    dest_dir = Path(args.dest_dir)
    if not dest_dir.is_absolute():
        dest_dir = ROOT / dest_dir

    run_tag = find_run_tag(source_dir, args.run_tag)
    copied_files = prepare_bundle(
        run_tag=run_tag,
        source_dir=source_dir,
        dest_dir=dest_dir,
        clean_dest=not args.no_clean,
    )

    print(f"Prepared production bundle for run_tag={run_tag}")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print("Copied files:")
    for rel_path in copied_files:
        print(f"- {rel_path}")


if __name__ == "__main__":
    main()
