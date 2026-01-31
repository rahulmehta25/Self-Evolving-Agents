"""
Artifact bundle export (Guide §6.4).

Produces a zip containing run_manifest.json, trace.jsonl, task_spec.json,
genome_spec.json, metrics_report.json, and logs/ for offline audit and sharing.
Optionally uploads the zip to a GCS bucket (set gcs_bucket or EVOAGENTBENCH_GCS_BUCKET).
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .data_store import DataStore


def _upload_to_gcs(local_path: str, bucket_name: str, object_name: str) -> str:
    """Upload a file to GCS. Returns gs://bucket/object_name. Uses ADC."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_path, content_type="application/zip")
    return f"gs://{bucket_name}/{object_name}"


def export_artifact_bundle(
    data_store: DataStore,
    run_id: str,
    out_dir: Optional[Path] = None,
    gcs_bucket: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Build and write the artifact bundle zip for a run.

    Contents (Guide §6.4):
      1. run_manifest.json
      2. trace.jsonl
      3. task_spec.json
      4. genome_spec.json
      5. metrics_report.json
      6. logs/

    Naming: artifact_bundle_{run_id}_{genome_id}_{task_id}.zip

    Args:
        data_store: DataStore instance to load run/trace/task/genome/metrics.
        run_id: Run ID.
        out_dir: Directory to write the zip into. Defaults to current working directory.
        gcs_bucket: If set, upload the zip to this GCS bucket (uses ADC).
            Can also be set via env EVOAGENTBENCH_GCS_BUCKET.

    Returns:
        (local_path, gcs_uri). gcs_uri is None if no bucket or upload not requested.
    """
    gcs_bucket = gcs_bucket or os.environ.get("EVOAGENTBENCH_GCS_BUCKET", "").strip() or None
    gs_uri: Optional[str] = None
    run_record = data_store.get_run(run_id)
    if not run_record:
        raise ValueError(f"Run not found: {run_id}")

    run_id_val = run_record.get("run_id") or run_id
    trace_id = run_record.get("trace_id")
    genome_id = run_record.get("genome_id", "")
    task_id = run_record.get("task_id", "")
    task_version = int(run_record.get("task_version") or 0)

    if not trace_id:
        raise ValueError(f"Run {run_id} has no trace_id")

    task = data_store.get_task(task_id, task_version)
    genome = data_store.get_genome(genome_id) if genome_id else None
    trace_events = data_store.get_trace(trace_id)
    metrics = data_store.get_metrics(run_id)

    manifest_raw = run_record.get("run_manifest_json")
    if isinstance(manifest_raw, str):
        try:
            run_manifest = json.loads(manifest_raw)
        except json.JSONDecodeError:
            run_manifest = {}
    else:
        run_manifest = dict(manifest_raw) if manifest_raw else {}

    out_dir = Path(out_dir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"artifact_bundle_{run_id_val}_{genome_id}_{task_id}.zip"
    zip_path = out_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run_manifest.json", json.dumps(run_manifest, indent=2))

        trace_lines = []
        for ev in trace_events:
            line = {
                "step_index": ev.get("step_index"),
                "timestamp": ev.get("timestamp"),
                "event_type": ev.get("event_type"),
                "payload": ev.get("payload"),
            }
            trace_lines.append(json.dumps(line))
        zf.writestr("trace.jsonl", "\n".join(trace_lines))

        zf.writestr("task_spec.json", json.dumps(task or {}, indent=2))
        zf.writestr("genome_spec.json", json.dumps(genome or {}, indent=2))
        zf.writestr("metrics_report.json", json.dumps(metrics, indent=2))
        # logs/ directory (empty placeholder; real logs can be added when available)
        zf.writestr("logs/.gitkeep", "")

    abs_path = str(zip_path.resolve())
    if gcs_bucket:
        object_name = f"artifacts/{zip_name}"
        gs_uri = _upload_to_gcs(abs_path, gcs_bucket, object_name)
    return (abs_path, gs_uri)
