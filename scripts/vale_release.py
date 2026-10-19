#!/usr/bin/env python3
"""Build the committed Vale package and check whether publication is needed.

This script only reads GitHub. The workflow owns the separate publication step.
"""

import argparse
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


def api_json(repository, path):
    url = f"https://api.github.com/repos/{repository}/{path}"
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"GitHub read failed with HTTP {error.code}") from None


def download_asset(repository, tag):
    # Public asset download; never forward the API token to the asset host.
    url = f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}/LanceDB.zip"
    with urlopen(url, timeout=30) as response:
        return response.read()


def package_contents(data):
    """Compare package files and permissions, not commit-dependent ZIP dates."""
    contents = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            name = entry.filename
            if (
                not name.startswith("LanceDB/")
                or ".." in name.split("/")
                or name in contents
            ):
                raise ValueError(f"Invalid or duplicate package entry: {name}")
            contents[name] = (entry.external_attr, archive.read(entry))
    if "LanceDB/meta.json" not in contents:
        raise ValueError("Package is missing LanceDB/meta.json")
    return contents


def existing_package_matches(repository, tag, archive):
    ref = api_json(repository, f"git/ref/tags/{quote(tag, safe='')}")
    release = api_json(repository, f"releases/tags/{quote(tag, safe='')}")
    if ref is None and release is None:
        return False
    if ref is None or release is None:
        raise RuntimeError(f"{tag} has an incomplete tag/release; inspect it before retrying")
    if release["draft"] or release["prerelease"] or release["tag_name"] != tag:
        raise RuntimeError(f"{tag} is not the expected published stable release")
    assets = [asset for asset in release["assets"] if asset["name"] == "LanceDB.zip"]
    if len(assets) != 1 or assets[0]["state"] != "uploaded":
        raise RuntimeError(f"{tag} does not have one fully uploaded LanceDB.zip")
    published = download_asset(repository, tag)
    if package_contents(published) != package_contents(archive.read_bytes()):
        raise RuntimeError(f"{tag} contains different package files; bump the package version")
    return True


def build_package(output_dir):
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    meta = json.loads(git("show", "HEAD:vale/LanceDB/meta.json"))
    version = meta["version"]
    if not isinstance(version, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version):
        raise ValueError("Expected a stable major.minor.patch package version")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "LanceDB.zip"
    subprocess.run(
        [
            "git", "archive", "--format=zip", "--prefix=LanceDB/",
            f"--mtime={git('show', '-s', '--format=%cI', 'HEAD')}",
            f"--output={archive}", "HEAD:vale/LanceDB",
        ],
        check=True,
    )
    return f"vale-lancedb-v{version}", archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "lancedb/docs"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("Expected owner/repository")
    tag, archive = build_package(args.output_dir.resolve())
    exists = existing_package_matches(args.repository, tag, archive)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"tag={tag}\nexists={str(exists).lower()}\n")
    print(f"{tag}: {'matching release exists; skip publication' if exists else 'ready for publication'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Vale release preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
