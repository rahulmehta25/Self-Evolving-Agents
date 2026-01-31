"""
GenomeValidator: Validates genome JSON before save/use (Guide §9).

Provides a single entry point for "is this genome valid?" so save/run
paths can reject invalid configs early. v1: basic JSON structure;
v2 can add stricter typed validation via schema_version.
"""

from typing import Any, Dict, Optional, Tuple

def _schema_path() -> Optional[str]:
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    p = base / "schemas" / "genome_spec_v1.json"
    if p.exists():
        return str(p)
    p = Path.cwd() / "evoagentbench" / "schemas" / "genome_spec_v1.json"
    if p.exists():
        return str(p)
    return None


def validate_genome(genome: Dict[str, Any], schema_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate a genome configuration. Use before saving or running (Guide §9).
    
    Args:
        genome: Genome dict (e.g. genome_id, system_prompt, llm_config).
        schema_path: Optional path to genome schema JSON; default auto-detected.
        
    Returns:
        (is_valid, error_message). error_message is None when valid.
    """
    if not isinstance(genome, dict):
        return False, "genome must be a dict"
    if not genome.get("genome_id") and "genome_id" not in genome:
        return False, "genome missing genome_id"
    if not isinstance(genome.get("system_prompt"), str):
        return False, "genome missing or invalid system_prompt"
    llm = genome.get("llm_config")
    if not isinstance(llm, dict):
        return False, "genome missing or invalid llm_config"
    if "model_name" not in llm and "temperature" not in llm:
        return False, "llm_config should include model_name and temperature"
    
    path = schema_path or _schema_path()
    if path:
        try:
            from evoagentbench.utils.validation import validate_genome_spec
            return validate_genome_spec(genome, path)
        except Exception:
            pass
    return True, None
