#!/usr/bin/env python3
"""
Run P1 SR across 3 seeds on remote.
"""
import subprocess, sys, time

seeds = [42, 123, 456]
for seed in seeds:
    print(f"\n{'='*60}")
    print(f"RUNNING SEED {seed}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, '-u', 'p1_sr.py', str(seed)],
        capture_output=True, text=True, timeout=7200
    )
    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    print(f"Completed in {elapsed:.0f}s")
    with open(f'p1_sr.sh', 'w') as f:
        f.write(f'#!/bin/bash\npython3 -u p1_sr.py {seed}\n')
