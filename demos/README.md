# Demos Directory (Unified Entry Points)

This directory provides stable import entry points for the full Marithmetics
ecosystem. Each file here is a lightweight wrapper that imports the actual
engine from its canonical directory (sm/, cosmo/, gr/, quantum/, controllers/, etc.)

This gives report generators, orchestration scripts, and notebooks a single,
consistent place to call:

    from demos.demo_sm_standard_model import main

without needing to know the internal repo layout.
