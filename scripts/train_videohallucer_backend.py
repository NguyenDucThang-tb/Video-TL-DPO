"""Compatibility entry point: all new runs use unified_config.md."""
from run_unified import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main("train"))
