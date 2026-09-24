from __future__ import annotations

import sys

from mc005_associative_lookup_response_marker_v9 import main


DEFAULTS = {
    "--model-id": "Qwen/Qwen3-0.6B",
    "--artifact-prefix": "mc005_qwen3_0p6b_associative_lookup_response_marker_v10",
    "--run-type": "associative_lookup_response_marker_v10_size_replication",
    "--progress-label": "response-atlas-v10",
    "--row-id-prefix": "mc005_v10",
}


def with_defaults(argv: list[str]) -> list[str]:
    args = list(argv)
    present = set()
    for index, value in enumerate(args):
        if value in DEFAULTS:
            present.add(value)
        if value.startswith("--") and "=" in value:
            present.add(value.split("=", 1)[0])
        if index > 0 and args[index - 1] in DEFAULTS:
            present.add(args[index - 1])
    prefixed: list[str] = []
    for flag, value in DEFAULTS.items():
        if flag not in present:
            prefixed.extend([flag, value])
    return prefixed + args


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *with_defaults(sys.argv[1:])]
    raise SystemExit(main())
