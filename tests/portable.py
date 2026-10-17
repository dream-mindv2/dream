#!/usr/bin/env python3
"""Compile once and run identical bytecode artifacts on each platform."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CASES = ["atoms", "bytes", "containers", "concurrency", "floats", "errors",
         "fusion", "deep_reads", "core_primitives"]


def run(command, **kwargs):
    if args.wine:
        # Wine exposes the host filesystem as Z:. Child VM arguments must also
        # be Windows paths; Wine only translates its executable argument.
        command = [args.wine, command[0], *[
            "Z:" + str(x) if isinstance(x, Path) and x.is_absolute() else x
            for x in command[1:]]]
    try:
        result = subprocess.run([str(x) for x in command], capture_output=True, timeout=180, **kwargs)
    except subprocess.TimeoutExpired as e:
        # What it printed before it stopped is where it stopped.
        out = (e.stdout or b"").decode(errors="replace")
        err = (e.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"{command}: timed out\n--- stdout ---\n{out}\n--- stderr ---\n{err}") from None
    if result.returncode:
        raise RuntimeError(f"{command}: exit {result.returncode}\n{result.stderr.decode(errors='replace')}")
    return result.stdout.replace(b"\r\n", b"\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["compile", "run"])
parser.add_argument("--wine-prefix", type=Path, help="Wine configuration directory")
parser.add_argument("--wine", type=Path, help="optional Wine executable for local Windows testing")
parser.add_argument("--vm", required=True, type=Path)
parser.add_argument("--images", required=True, type=Path)
args = parser.parse_args()
vm = args.vm.resolve()
images = args.images.resolve()
if args.wine:
    os.environ.setdefault("WINEPREFIX", str(args.wine_prefix.resolve() if args.wine_prefix else images.parent / "wine"))
    os.environ.setdefault("WINEDEBUG", "-all")
    os.environ["DISPLAY"] = ""
if args.mode == "compile":
    images.mkdir(parents=True, exist_ok=True)
    compiler = ROOT / "dreams/bootstrap/dreams.dream"
    cases = []
    for name in CASES:
        source = ROOT / "dream/tests/programs" / f"{name}.dr"
        if not source.exists():
            raise RuntimeError(f"Missing portability fixture: {source}")
        output = images / f"{name}.dream"
        run([vm, compiler, "-L", ROOT / "mind", source, "-o", output])
        expected = source.with_suffix(".expected").read_bytes().replace(b"\r\n", b"\n")
        cases.append(dict(name=name, sha256=digest(output), expected=expected.decode()))
    run([vm, compiler, "-L", ROOT / "mind", ROOT / "dream/tests/portable.dr",
         "-o", images / "portable.dream"])
    cases.append(dict(name="portable", sha256=digest(images / "portable.dream")))
    run([vm, compiler, "-L", ROOT / "mind", "-L", ROOT, ROOT / "dream/tests/portable_paths.dr",
         "-o", images / "paths.dream"])
    cases.append(dict(name="paths", sha256=digest(images / "paths.dream"), expected="true\n" * 7))
    # std.tls through whichever library this platform's VM was built with:
    # OpenSSL on Linux and macOS, SChannel on Windows. One image, one output.
    run([vm, compiler, "-L", ROOT / "mind", ROOT / "dream/tests/tls/tls.dr", "-o", images / "tls.dream"])
    cases.append(dict(name="tls", sha256=digest(images / "tls.dream")))
    run([vm, compiler, "-L", ROOT / "mind", "-L", ROOT, ROOT / "dreams/src/main.dr",
         "-o", images / "dreams.dream"])
    cases.append(dict(name="dreams", sha256=digest(images / "dreams.dream")))
    (images / "manifest.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
else:
    cases = json.loads((images / "manifest.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="dream portable ") as scratch:
        for case in cases:
            image = images / (case["name"] + ".dream")
            assert digest(image) == case["sha256"], f"Image changed: {image}"
            if case["name"] == "dreams":
                # The very same compiler image must compile on the receiving OS.
                output = Path(scratch) / "compiled.dream"
                run([vm, image, "-L", ROOT / "mind", ROOT / "dream/tests/programs/bytes.dr", "-o", output])
                expected = (ROOT / "dream/tests/programs/bytes.expected").read_bytes().replace(b"\r\n", b"\n")
                assert run([vm, output]).rstrip(b"\n") == expected.rstrip(b"\n")
            elif case["name"] == "tls":
                got = run([vm, image, ROOT / "dream/tests/tls"])
                assert got.rstrip(b"\n").endswith(b"all tests passed"), got.decode(errors="replace")
            elif case["name"] == "portable":
                got = run([vm, "--workers", "1", image, vm, image, Path(scratch) / "bytes ü.dat"])
                platform = 'windows' if args.wine or os.name == 'nt' else ('macos' if sys.platform == 'darwin' else 'linux')
                expected = (':' + platform + '\n').encode() + b'true\ntrue\ntrue\n["", "two words", "quote\\"here", "trail\\\\"]\n\ntrue\ntrue\n'
                assert got == expected, (got, expected)
            else:
                for options in (["--no-jit"], []):
                    got = run([vm, *options, image])
                    assert got.rstrip(b"\n") == case["expected"].encode().rstrip(b"\n"), (image, got)
            print(f"PASS {case['name']}", flush=True)
