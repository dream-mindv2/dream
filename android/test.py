#!/usr/bin/env python3
"""Run the unchanged portable image bundle on an adb-connected Android device."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="selected",
                        help="run only this named manifest case (repeatable)")
    args = parser.parse_args()
    adb = [args.adb, "-s", args.serial]
    remote = "/data/local/tmp/dream-" + uuid.uuid4().hex

    def call(command):
        result = subprocess.run(command, capture_output=True, timeout=600)
        if result.returncode:
            raise RuntimeError(f"{command}: exit {result.returncode}\n"
                               + result.stdout.decode(errors="replace")
                               + result.stderr.decode(errors="replace"))
        return result.stdout.replace(b"\r\n", b"\n")

    def shell(*words):
        return call([*adb, "shell", shlex.join(words)])

    def vm(*words):
        return shell("env", "LD_LIBRARY_PATH=" + remote, "TMPDIR=" + remote,
                     "DREAM_CORES=2", remote + "/dream", *map(str, words))

    cases = json.loads((args.images / "manifest.json").read_text())
    if args.selected:
        unknown = set(args.selected) - {case["name"] for case in cases}
        if unknown:
            parser.error("unknown cases: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["name"] in args.selected]
    for case in cases:
        image = args.images / (case["name"] + ".dream")
        assert hashlib.sha256(image.read_bytes()).hexdigest() == case["sha256"], image
    shell("mkdir", "-p", remote)
    try:
        for name in ["dream", "dream_tests", "libdream.so", "libc++_shared.so"]:
            call([*adb, "push", str(args.dist / name), remote + "/" + name])
        for directory, name in [(args.images, "images"), (ROOT / "mind", "mind"),
                                (ROOT / "dream/tests/tls", "tls")]:
            call([*adb, "push", str(directory), remote + "/" + name])
        shell("chmod", "755", remote + "/dream", remote + "/dream_tests")
        shell("env", "LD_LIBRARY_PATH=" + remote, "TMPDIR=" + remote,
              "DREAM_CORES=2", remote + "/dream_tests")
        print("PASS VM unit tests", flush=True)
        for case in cases:
            image = remote + "/images/" + case["name"] + ".dream"
            actual_hash = shell("sha256sum", image).decode().split()[0]
            assert actual_hash == case["sha256"], (image, actual_hash)
            if case["name"] == "dreams":
                source = ROOT / "dream/tests/programs/bytes.dr"
                call([*adb, "push", str(source), remote + "/bytes.dr"])
                output = remote + "/compiled.dream"
                vm(image, "-L", remote + "/mind", remote + "/bytes.dr", "-o", output)
                assert vm(output).rstrip(b"\n") == source.with_suffix(".expected").read_bytes().rstrip(b"\n")
                # Cross-target checks go both ways: Android accepts its own
                # image and rejects a Linux image before running it.
                vm(image, "-L", remote + "/mind", remote + "/bytes.dr",
                   "--target", "android", "-o", output)
                vm(output)
                vm(image, "-L", remote + "/mind", remote + "/bytes.dr",
                   "--target", "linux", "-o", output)
                try:
                    vm(output)
                except RuntimeError as error:
                    assert "was built for linux" in str(error), error
                else:
                    raise AssertionError("Android accepted a Linux image")
                call([*adb, "push", str(ROOT / "dream/android/smoke.dr"), remote + "/smoke.dr"])
                vm(image, "-L", remote + "/mind", remote + "/smoke.dr", "-o", output)
                assert vm(output) == b"true\ntrue\ntrue\n"
            elif case["name"] == "tls":
                got = vm(image, remote + "/tls")
                assert got.rstrip().endswith(b"all tests passed"), got
            elif case["name"] == "portable":
                got = vm("--workers", "1", image, remote + "/dream", image,
                         remote + "/bytes ü.dat")
                expected = (b':android\ntrue\ntrue\ntrue\n'
                            b'["", "two words", "quote\\"here", "trail\\\\"]\n\ntrue\ntrue\n')
                assert got == expected, (got, expected)
            else:
                got = vm(image)
                assert got.rstrip(b"\n") == case["expected"].encode().rstrip(b"\n"), (image, got)
            print("PASS " + case["name"], flush=True)
    finally:
        shell("rm", "-rf", remote)


if __name__ == "__main__":
    main()
