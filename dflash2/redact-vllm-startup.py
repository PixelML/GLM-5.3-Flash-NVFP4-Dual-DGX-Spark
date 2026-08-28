"""Redact vLLM's API key from its non-default-arguments startup log."""

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py"
)
source = path.read_text()
old = """def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info(\"non-default args: %s\", non_default_args)
"""
new = """def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    if \"api_key\" in non_default_args:
        non_default_args[\"api_key\"] = [\"***REDACTED***\"]
    logger.info(\"non-default args: %s\", non_default_args)
"""
if old not in source:
    raise SystemExit("vLLM startup-log patch anchor not found")
path.write_text(source.replace(old, new, 1))
