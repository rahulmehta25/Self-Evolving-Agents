"""
LLM adapters for the agent runner.

Supports mock (default) and Vertex AI Gemini. Set EVOAGENTBENCH_LLM=vertex
to use Gemini; use EVOAGENTBENCH_GCP_PROJECT and EVOAGENTBENCH_GCP_LOCATION
to override project/location (defaults from ADC and us-central1).
Guide §5.2: retries with exponential backoff, ExternalFailure when exhausted.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple


class ExternalFailureError(Exception):
    """Raised when LLM/tool external calls fail after retries (Guide §5.2)."""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause

_PROJECT = os.environ.get("EVOAGENTBENCH_GCP_PROJECT", "")
_LOCATION = os.environ.get("EVOAGENTBENCH_GCP_LOCATION", "us-central1")


def _vertex_gemini_available() -> bool:
    try:
        import vertexai  # noqa: F401
        from vertexai.generative_models import GenerativeModel  # noqa: F401
        return True
    except ImportError:
        return False


def call_vertex_gemini(
    system_prompt: str,
    user_prompt: str,
    llm_config: Dict[str, Any],
    run_seed: int,
) -> Tuple[str, int, int]:
    """
    Call Vertex AI Gemini. Uses ADC for credentials.

    Args:
        system_prompt: System instruction for the model.
        user_prompt: User message content.
        llm_config: Genome llm_config (model_name, temperature, top_p, max_tokens).
        run_seed: Seed for reproducibility (used as Gemini seed when supported).

    Returns:
        (response_text, input_tokens, output_tokens)
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    project = _PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        # Try to derive from ADC / gcloud
        try:
            import subprocess
            out = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                project = out.stdout.strip()
        except Exception:
            pass
    if not project:
        raise ValueError(
            "Vertex AI requires a GCP project. Set EVOAGENTBENCH_GCP_PROJECT or "
            "GOOGLE_CLOUD_PROJECT, or run: gcloud config set project YOUR_PROJECT"
        )

    vertexai.init(project=project, location=_LOCATION)

    model_id = str(llm_config.get("model_name") or "gemini-1.5-flash").strip()
    if not model_id.startswith("gemini-"):
        model_id = f"gemini-1.5-flash"  # fallback

    temperature = float(llm_config.get("temperature", 0.0))
    top_p = float(llm_config.get("top_p", 1.0))
    max_tokens = int(llm_config.get("max_tokens", 2048))

    kwargs = dict(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    if run_seed is not None and run_seed != 0:
        kwargs["seed"] = run_seed
    gen_config = GenerationConfig(**kwargs)

    model = GenerativeModel(
        model_id,
        system_instruction=[system_prompt] if system_prompt else None,
        generation_config=gen_config,
    )

    # Guide §5.2: up to 3 tries, exponential backoff; ExternalFailure when exhausted
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = model.generate_content(user_prompt or " ")
            if not response or not response.candidates:
                return ("", 0, 0)
            text = (response.candidates[0].content.parts[0].text or "") if response.candidates[0].content.parts else ""
            usage = response.usage_metadata or {}
            inp = int(usage.get("prompt_token_count", 0))
            out = int(usage.get("candidates_token_count", 0))
            return (text, inp, out)
        except Exception as e:
            last_err = e
            if attempt < 2:
                delay = 2 ** attempt
                time.sleep(delay)
            else:
                raise ExternalFailureError(
                    f"Vertex AI call failed after 3 attempts: {e}",
                    cause=last_err,
                ) from last_err
    raise ExternalFailureError("Vertex AI call failed", cause=last_err)


def call_vertex_gemini_text_only(
    prompt: str,
    model_id: str = "gemini-1.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Single-prompt Vertex Gemini call (e.g. for Judge). Uses ADC.
    Returns raw response text. Uses EVOAGENTBENCH_GCP_PROJECT / EVOAGENTBENCH_GCP_LOCATION.
    """
    if not _vertex_gemini_available():
        raise RuntimeError("Vertex AI SDK not available")
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    project = _PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT")
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
    if not project:
        raise ValueError("Set EVOAGENTBENCH_GCP_PROJECT or gcloud config set project")
    vertexai.init(project=project, location=_LOCATION)
    model = GenerativeModel(model_id, generation_config=GenerationConfig(
        temperature=temperature, max_output_tokens=max_tokens,
    ))
    response = model.generate_content(prompt or " ")
    if not response or not response.candidates:
        return ""
    if response.candidates[0].content.parts:
        return (response.candidates[0].content.parts[0].text or "").strip()
    return ""


def get_llm_provider() -> str:
    """Return 'vertex' or 'mock'. Default is vertex when Vertex is available and env is unset."""
    v = (os.environ.get("EVOAGENTBENCH_LLM") or "").strip().lower()
    if v == "mock":
        return "mock"
    if v == "vertex":
        return "vertex"
    # Default: use Vertex if the SDK is available (GCP is set up)
    if _vertex_gemini_available():
        return "vertex"
    return "mock"
