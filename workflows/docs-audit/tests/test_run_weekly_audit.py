from __future__ import annotations

from datetime import datetime, timezone

from scripts.run_weekly_audit import debug_finding_rows


def test_debug_finding_rows_serializes_datetime_and_strips_embedding() -> None:
    rows = [
        {
            "id": "run:001",
            "completed_at": datetime(2026, 5, 18, tzinfo=timezone.utc),
            "embedding": [1.0, 2.0],
        }
    ]

    assert debug_finding_rows(rows) == [
        {
            "id": "run:001",
            "completed_at": "2026-05-18T00:00:00+00:00",
            "embedding": [],
        }
    ]
