from __future__ import annotations

from docs_audit.deterministic_runner import RepoInfo, repo_snapshot


def test_repo_snapshot_skips_missing_repo_path(tmp_path) -> None:
    missing_repo = tmp_path / "lancedb"

    snapshot = repo_snapshot(
        RepoInfo(name="lancedb", path=missing_repo),
        refresh=True,
        simulate_failure=False,
    )

    assert snapshot == {
        "repo": "lancedb",
        "path": str(missing_repo),
        "branch": "",
        "sha_before": "",
        "dirty": False,
        "refresh_requested": True,
        "refresh_status": "skipped",
        "message": "Configured repo path does not exist",
    }


def test_repo_snapshot_skips_non_git_repo_path(tmp_path) -> None:
    non_git_repo = tmp_path / "sophon"
    non_git_repo.mkdir()

    snapshot = repo_snapshot(
        RepoInfo(name="sophon", path=non_git_repo),
        refresh=False,
        simulate_failure=False,
    )

    assert snapshot == {
        "repo": "sophon",
        "path": str(non_git_repo),
        "branch": "",
        "sha_before": "",
        "dirty": False,
        "refresh_requested": False,
        "refresh_status": "skipped",
        "message": "Configured repo path is not a git checkout",
    }
