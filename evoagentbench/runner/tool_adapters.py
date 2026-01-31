"""
Tool adapters: Vertex AI Search (retrieval) and configurable real API URLs.

- Vertex Search: set EVOAGENTBENCH_VERTEX_SEARCH_DATA_STORE (data store ID);
  EVOAGENTBENCH_GCP_PROJECT and EVOAGENTBENCH_GCP_LOCATION (default global).
- Real API: set EVOAGENTBENCH_TOOL_<TOOL_NAME>_URL (e.g. EVOAGENTBENCH_TOOL_GET_WEATHER_URL).
  Runner POSTs {"arguments": tool_args} and uses response JSON as result.
"""

import json
import os
from typing import Any, Dict, List, Optional


def _get_project_location() -> tuple:
    project = os.environ.get("EVOAGENTBENCH_GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        try:
            import subprocess
            out = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                project = out.stdout.strip()
        except Exception:
            pass
    location = os.environ.get("EVOAGENTBENCH_GCP_LOCATION", "global")
    return (project or "", location)


def vertex_search_available() -> bool:
    """True if Vertex Search data store is configured and SDK available."""
    if not (os.environ.get("EVOAGENTBENCH_VERTEX_SEARCH_DATA_STORE") or "").strip():
        return False
    try:
        from google.cloud import discoveryengine_v1
        return True
    except ImportError:
        return False


def call_vertex_search(query: str, page_size: int = 5) -> List[Dict[str, Any]]:
    """
    Call Vertex AI Search (Discovery Engine). Returns list of {text, source_id} for citation checker.
    Uses EVOAGENTBENCH_VERTEX_SEARCH_DATA_STORE, EVOAGENTBENCH_GCP_PROJECT, EVOAGENTBENCH_GCP_LOCATION.
    """
    from google.cloud import discoveryengine_v1 as discoveryengine

    data_store_id = (os.environ.get("EVOAGENTBENCH_VERTEX_SEARCH_DATA_STORE") or "").strip()
    if not data_store_id:
        return []
    project, location = _get_project_location()
    if not project:
        return []

    client = discoveryengine.SearchServiceClient()
    serving_config = client.serving_config_path(
        project=project,
        location=location,
        data_store=data_store_id,
        serving_config="default_serving_config",
    )
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=page_size,
    )
    results: List[Dict[str, Any]] = []
    try:
        pager = client.search(request=request)
        for response in pager:
            for r in getattr(response, "results", []) or []:
                source_id = getattr(r, "id", None) or ""
                text = ""
                doc = getattr(r, "document", None)
                if doc:
                    if hasattr(doc, "derived_struct_data") and doc.derived_struct_data:
                        d = doc.derived_struct_data
                        if isinstance(d, dict):
                            sn = (d.get("snippets") or [{}])[0]
                            text = (sn.get("snippet") or {}).get("content", "") if isinstance(sn, dict) else ""
                            if not text:
                                text = str(d.get("link", "")) or json.dumps(d)[:500]
                    if not text and hasattr(doc, "id"):
                        text = str(doc.id)
                results.append({"source_id": str(source_id), "text": text or str(source_id)})
    except Exception:
        results = []
    return results


def get_tool_url(tool_name: str) -> Optional[str]:
    """Get configured URL for a tool from EVOAGENTBENCH_TOOL_<NAME>_URL."""
    key = "EVOAGENTBENCH_TOOL_" + tool_name.upper().replace("-", "_") + "_URL"
    url = (os.environ.get(key) or "").strip()
    return url or None


def call_real_tool_url(tool_name: str, arguments: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """
    POST to the configured tool URL with {"tool_name": name, "arguments": arguments}; return response JSON.
    """
    import requests

    url = get_tool_url(tool_name)
    if not url:
        return {}
    try:
        resp = requests.post(
            url,
            json={"tool_name": tool_name, "arguments": arguments},
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except Exception:
        return {}
