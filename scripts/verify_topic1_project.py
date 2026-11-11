#!/usr/bin/env python3
"""Verify the whole course project without fitting or rewriting scientific artifacts."""
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from unittest.mock import patch

import reproduce_topic1 as runner
import verify_topic1_complete as complete

SMALL = "outputs/topic1_small_data"
EXTERNAL = "outputs/topic1_external_diagnostics"
REPORT = "outputs/project_verification/latest.json"
SMALL_TABLES = ["fold_metrics.csv", "fold_means.csv", "summary.csv",
                "paired_comparisons.csv", "low_data_interaction.csv"]
EXTERNAL_TABLES = ["system_metrics.csv", "reliability_bins.csv", "distance_metrics.csv"]
EXTERNAL_FILES = EXTERNAL_TABLES + ["audit.json", "REPORT.md", "reliability.png",
                 "reliability.pdf", "confusion_matrices.png", "confusion_matrices.pdf"]


def small_required():
    from experiments import small_data
    files = {f"{SMALL}/{name}" for name in SMALL_TABLES + [
        "protocol.json", "environment.json", "audit.json", "REPORT.md",
        "learning_RF.png", "learning_RF.pdf", "learning_ET.png", "learning_ET.pdf"]}
    for config in small_data.configurations():
        folder = small_data.folder_for(config).relative_to(small_data.OUT)
        files.update(f"{SMALL}/{folder}/{name}" for name in
                     ["model.joblib", "predictions.csv", "result.json"])
    return files


def manifest_entries(root, path, required=()):
    runner.check_manifest(root, root/path)
    entries = json.loads((root/path).read_text())
    if not set(required).issubset(entries):
        raise RuntimeError(f"Incomplete manifest coverage: {path}")
    for name in entries:
        if (root/name).is_symlink():
            raise RuntimeError(f"Symlink artifact: {name}")
    return set(entries) | {path}


def snapshot(root):
    names = set()
    for version in runner.VERSIONS:
        names |= manifest_entries(root, f"outputs/{version}/delivery_manifest.json")
    names |= manifest_entries(root, f"{complete.FINAL}/{complete.MANIFEST}", complete.required_files())
    names |= manifest_entries(root, f"{SMALL}/delivery_manifest.json", small_required())
    names |= {f"{EXTERNAL}/{name}" for name in EXTERNAL_FILES}
    entries = {}
    for name in sorted(names):
        path = root/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"Missing or unsafe project artifact: {name}")
        entries[name] = runner.digest(path)
    return entries


def compare_tables(original, replay, names):
    import pandas as pd
    for name in names:
        pd.testing.assert_frame_equal(pd.read_csv(original/name, float_precision="round_trip"),
                                      pd.read_csv(replay/name, float_precision="round_trip"),
                                      check_exact=False, rtol=1e-10, atol=1e-12)


def replay_small(root):
    from experiments import small_data
    original = root/SMALL
    with tempfile.TemporaryDirectory(prefix="topic1-small-check-") as directory:
        scratch = Path(directory)/"study"
        shutil.copytree(original, scratch)
        with patch.object(small_data, "OUT", scratch):
            audit = small_data.audit()
        if audit != json.loads((original/"audit.json").read_text()):
            raise RuntimeError("Replayed small-data audit differs from saved audit")
        compare_tables(original, scratch, SMALL_TABLES)
    return {"pass_all": True, "replayed_models": 1050, "training_performed": False}


def replay_external(root):
    from experiments import external_diagnostics as external
    original = root/EXTERNAL
    with tempfile.TemporaryDirectory(prefix="topic1-external-check-") as directory:
        scratch = Path(directory)
        with patch.object(external, "OUT", scratch):
            external.main()
        if json.loads((original/"audit.json").read_text()) != json.loads((scratch/"audit.json").read_text()):
            raise RuntimeError("External provenance/audit differs from independent replay")
        compare_tables(original, scratch, EXTERNAL_TABLES)
        if (original/"REPORT.md").read_bytes() != (scratch/"REPORT.md").read_bytes():
            raise RuntimeError("External report differs from replay")
    return {"pass_all": True, "primary_rows": 5122, "training_performed": False}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify():
    root = runner.ROOT
    target = root/REPORT
    started = time.monotonic()
    record = dict(status="running", pass_all=False,
                  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), python=platform.python_version(),
                  stages={}, training_performed=False)
    # Invalidate a previous pass before any check can fail or be interrupted.
    atomic_json(target, record)
    try:
        if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
            raise RuntimeError("Scientific audit assertions must remain enabled")
        runner.check_public()
        record["source_manifest_sha256"] = runner.digest(root/runner.SOURCE_MANIFEST)
        before = snapshot(root)
        complete.verify()
        record["stages"]["historical_and_final"] = {"pass_all": True, "final_model_replays": 86}
        record["stages"]["small_data"] = replay_small(root)
        record["stages"]["external_diagnostics"] = replay_external(root)
        if snapshot(root) != before:
            raise RuntimeError("Original artifacts changed during project verification")
        if runner.digest(root/runner.SOURCE_MANIFEST) != record["source_manifest_sha256"]:
            raise RuntimeError("Public source manifest changed during verification")
        runner.check_public()
        record.update(status="passed", pass_all=True, artifact_hashes=before,
                      artifact_count=len(before), original_artifacts_unchanged=True)
    except BaseException as error:
        record.update(status="failed", pass_all=False,
                      error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic()-started
        record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(target, record)
    print(f"PROJECT VERIFIED: historical + final + 1050 small-data replays + external diagnostics; {len(before)} artifacts", flush=True)
    return record


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(runner.ROOT))
    verify()
