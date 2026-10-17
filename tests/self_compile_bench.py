#!/usr/bin/env python3
"""Time real self-compiles, checking the bootstrap fixpoint on every run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vm', type=Path, default=root / 'build-dream/bin/dream')
    parser.add_argument('--seed', type=Path, default=root / 'dreams/bootstrap/dreams.dream')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--no-jit', action='store_true')
    parser.add_argument('--target', type=float, default=1.25)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.runs < 5 or args.workers < 0 or args.target <= 0 or args.timeout <= 0:
        parser.error('require at least 5 runs, nonnegative workers, and positive target/timeout')
    vm, seed = args.vm.resolve(), args.seed.resolve()
    command = [str(vm)]
    if args.no_jit:
        command.append('--no-jit')
    if args.workers:
        command += ['-j', str(args.workers)]
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in
                    Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), cpu)
    cache = vm.parent.parent / 'CMakeCache.txt'
    build = [line for line in cache.read_text().splitlines()
             if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:',
                                 'DREAM_PGO:', 'CMAKE_CXX_FLAGS:',
                                 'CMAKE_INTERPROCEDURAL_OPTIMIZATION:'))] if cache.exists() else []
    report = dict(vm=str(vm), cpu=cpu, build=build, workers=args.workers or 'default',
                  jit=not args.no_jit, target=args.target,
                  seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                  environment={k: v for k, v in os.environ.items()
                               if k.startswith(('DREAM_', 'MIND_'))})
    with tempfile.TemporaryDirectory(prefix='dream-self-compile-') as temp:
        stage2, output = Path(temp) / 'stage2.dream', Path(temp) / 'output.dream'

        def compile_image(image, dest):
            dest.unlink(missing_ok=True)
            start = time.perf_counter()
            run = subprocess.run(command + [str(image), '-L', 'mind', '-L', '.',
                                 '-o', str(dest), 'dreams/src/main.dr'], cwd=root,
                                 capture_output=True, text=True, timeout=args.timeout)
            elapsed = time.perf_counter() - start
            if run.returncode:
                raise RuntimeError(f'compiler exited {run.returncode}:\n{run.stdout}{run.stderr}')
            return elapsed

        # First invocation, not a claim that OS filesystem caches are cold.
        report['seed_first_seconds'] = compile_image(seed, stage2)
        expected = stage2.read_bytes()
        report['image_sha256'] = hashlib.sha256(expected).hexdigest()
        report['warmup_seconds'] = compile_image(stage2, output)
        if output.read_bytes() != expected:
            raise RuntimeError('bootstrap fixpoint differs during warmup')
        samples = []
        for _ in range(args.runs):
            samples.append(compile_image(stage2, output))
            if output.read_bytes() != expected:
                raise RuntimeError('self-compile output differs')
        report.update(seconds=samples, median=statistics.median(samples), maximum=max(samples),
                      passed=all(sample < args.target for sample in samples))
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
