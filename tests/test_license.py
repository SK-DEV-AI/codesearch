"""The licence field must carry a name, not a licence body."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libraries_io import _license_name  # noqa: E402

BSD_BODY = """Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice
"""


def test_short_spdx_name_survives() -> None:
    assert _license_name("MIT") == "MIT"
    assert _license_name("Apache-2.0") == "Apache-2.0"
    assert _license_name("BSD-3-Clause") == "BSD-3-Clause"


def test_empty_and_missing_values_stay_empty() -> None:
    assert _license_name("") == ""
    assert _license_name(None) == ""
    assert _license_name(42) == ""


def test_body_with_a_named_first_line_keeps_the_name() -> None:
    out = _license_name("MIT License\n\nCopyright (c) 2026 Someone\n")
    assert out == "MIT License (license text omitted)"


def test_body_without_a_name_is_reported_as_dropped() -> None:
    assert _license_name(BSD_BODY) == "license text omitted (unnamed)"


def test_the_body_never_leaks_into_the_field() -> None:
    # The whole point: 1.5 KB of legal text must not reach the caller.
    for raw in (BSD_BODY, "MIT License\n" + BSD_BODY * 5):
        out = _license_name(raw)
        assert len(out) < 80
        assert "Redistribution" not in out
