"""Release checks are read-only; all GitHub responses here are stand-ins."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile

spec = importlib.util.spec_from_file_location(
    "vale_release", Path(__file__).resolve().parents[1] / "vale_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def archive(content=b"rule", year=2026, mode=0o100644, extra=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        entries = [("LanceDB/meta.json", b'{"version":"0.1.0"}'),
                   ("LanceDB/styles/LanceDB/Rule.yml", content)]
        if extra:
            entries.append(("LanceDB/unexpected.txt", b"extra"))
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=(year, 1, 1, 0, 0, 0))
            info.external_attr = mode << 16
            output.writestr(info, data)
    return buffer.getvalue()


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "LanceDB.zip"
        self.path.write_bytes(archive())
        self.ref = {"object": {"type": "commit", "sha": "a" * 40}}
        self.metadata = {
            "draft": False, "prerelease": False, "tag_name": "vale-lancedb-v0.1.0",
            "assets": [{"name": "LanceDB.zip", "state": "uploaded"}],
        }

    def check(self, ref, metadata, data=None):
        with patch.object(release, "api_json", side_effect=[ref, metadata]), patch.object(
            release, "download_asset", return_value=data or archive()
        ) as download:
            result = release.existing_package_matches(
                "lancedb/docs", "vale-lancedb-v0.1.0", self.path
            )
            if metadata is None:
                download.assert_not_called()
            return result

    def test_absent_release_can_publish(self):
        self.assertFalse(self.check(None, None))

    def test_matching_release_skips_even_with_different_zip_dates(self):
        self.assertTrue(self.check(self.ref, self.metadata, archive(year=2025)))

    def test_different_rule_requires_new_version(self):
        with self.assertRaisesRegex(RuntimeError, "bump the package version"):
            self.check(self.ref, self.metadata, archive(content=b"different"))

    def test_extra_file_requires_new_version(self):
        with self.assertRaisesRegex(RuntimeError, "different package files"):
            self.check(self.ref, self.metadata, archive(extra=True))

    def test_changed_permissions_require_new_version(self):
        with self.assertRaisesRegex(RuntimeError, "different package files"):
            self.check(self.ref, self.metadata, archive(mode=0o100755))

    def test_incomplete_release_stops(self):
        for ref, metadata in [(self.ref, None), (None, self.metadata)]:
            with self.subTest(ref=ref), self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.check(ref, metadata)

    def test_unpublished_or_wrong_release_stops(self):
        for key, value in [("draft", True), ("prerelease", True), ("tag_name", "wrong")]:
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "stable release"):
                self.check(self.ref, {**self.metadata, key: value})

    def test_missing_duplicate_or_pending_asset_stops(self):
        for assets in [[], self.metadata["assets"] * 2, [{"name": "LanceDB.zip", "state": "new"}]]:
            with self.subTest(assets=assets), self.assertRaisesRegex(RuntimeError, "fully uploaded"):
                self.check(self.ref, {**self.metadata, "assets": assets})

    def test_corrupt_zip_stops(self):
        with self.assertRaises(zipfile.BadZipFile):
            self.check(self.ref, self.metadata, b"not a zip")

    def test_duplicate_and_outside_paths_stop(self):
        for names in [["LanceDB/meta.json"] * 2, ["outside"], ["LanceDB/../escape"]]:
            with self.subTest(names=names):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as output:
                    for name in names:
                        # zipfile warns for the intentional duplicate fixture.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            output.writestr(name, b"{}")
                with self.assertRaises(ValueError):
                    release.package_contents(buffer.getvalue())

    def test_api_only_treats_404_as_absent(self):
        for status in [404, 401, 403, 429, 500]:
            with self.subTest(status=status), patch.object(
                release, "urlopen", side_effect=HTTPError("https://api.github.com/", status, "failure", {}, None)
            ):
                if status == 404:
                    self.assertIsNone(release.api_json("lancedb/docs", "releases/tags/test"))
                else:
                    with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                        release.api_json("lancedb/docs", "releases/tags/test")

    def test_asset_download_does_not_forward_token(self):
        with patch.dict(release.os.environ, {"GH_TOKEN": "test-only-token"}), patch.object(release, "urlopen") as request:
            release.download_asset("lancedb/docs", "vale-lancedb-v0.1.0")
            self.assertIsInstance(request.call_args.args[0], str)
            self.assertEqual(request.call_args.kwargs, {"timeout": 30})

    def test_invalid_version_stops_before_build(self):
        for version in ["bad", "01.2.3", "1.0.0-beta.1", 1]:
            with self.subTest(version=version), patch.object(
                release.subprocess, "check_output", return_value=json.dumps({"version": version})
            ), patch.object(release.subprocess, "run") as build:
                with self.assertRaises(ValueError):
                    release.build_package(Path(self.directory.name))
                build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
