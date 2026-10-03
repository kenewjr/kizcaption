"""Update checker for KizCaption using GitHub Releases API."""
from __future__ import annotations

import json
import re
from typing import Any
import urllib.request

REPO_OWNER = "kenewjr"
REPO_NAME = "kizcaption"
GITHUB_API_LATEST = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
DEFAULT_RELEASE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases"


def parse_version(ver_str: str) -> tuple[int, ...]:
    """Parse version string like 'v1.0.0' or '1.2.3' into a comparable integer tuple."""
    nums = re.findall(r"\d+", str(ver_str))
    return tuple(map(int, nums)) if nums else (0,)


def check_for_updates(current_version: str, timeout: float = 6.0) -> dict[str, Any]:
    """Check GitHub releases for newer version.

    Returns dict:
    - has_update: bool
    - latest_version: str
    - release_url: str
    - release_notes: str
    - error: str | None
    """
    req = urllib.request.Request(
        GITHUB_API_LATEST,
        headers={
            "User-Agent": f"KizCaption/{current_version} (Windows)",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return {
                    "has_update": False,
                    "latest_version": current_version,
                    "release_url": DEFAULT_RELEASE_URL,
                    "release_notes": "",
                    "error": f"HTTP {resp.status}",
                }
            data = json.loads(resp.read().decode("utf-8"))
            raw_tag = str(data.get("tag_name", "")).strip()
            clean_tag = raw_tag.lstrip("vV") or current_version
            latest_v = parse_version(clean_tag)
            current_v = parse_version(current_version)
            has_update = latest_v > current_v

            return {
                "has_update": has_update,
                "latest_version": clean_tag,
                "release_url": str(data.get("html_url", DEFAULT_RELEASE_URL)),
                "release_notes": str(data.get("body", "")),
                "error": None,
            }
    except Exception as exc:
        return {
            "has_update": False,
            "latest_version": current_version,
            "release_url": DEFAULT_RELEASE_URL,
            "release_notes": "",
            "error": str(exc),
        }
