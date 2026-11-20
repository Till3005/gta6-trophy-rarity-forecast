"""Dev convenience shim: delegates to bot/main.py (the contract entrypoint)."""

import os
import runpy

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot", "main.py"),
    run_name="__main__",
)
