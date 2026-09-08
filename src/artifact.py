"""Tiện ích đọc và kiểm tra release bundle bất biến."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_RELEASE_FILES: tuple[str, ...] = (
    "model.joblib",
    "model_config.json",
    "feature_contract.json",
    "calibration.json",
    "decision_policy.json",
    "reference_distribution.json",
    "data_manifest.json",
    "validation_metrics.json",
    "locked_test_metrics.json",
    "MODEL_CARD.md",
    "manifest.json",
)


def sha256_file(path: str | Path) -> str:
    """Tính SHA256 theo nội dung file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest_release(releases_dir: str | Path = "releases") -> Path | None:
    """Tìm release mới nhất có manifest, không suy đoán từ thư mục legacy."""
    root = Path(releases_dir)
    if not root.exists():
        return None
    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def verify_release_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Kiểm tra đủ file, feature contract và hash của toàn bộ release."""
    root = Path(bundle_dir)
    result: dict[str, Any] = {"bundle_exists": root.exists(), "hashes_match": False}
    if not root.exists():
        result["error"] = "Không tìm thấy release bundle."
        return result

    missing = [name for name in REQUIRED_RELEASE_FILES if not (root / name).exists()]
    result["missing_files"] = missing
    if missing:
        result["error"] = "Release bundle thiếu file bắt buộc."
        return result

    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        expected_hashes = manifest.get("file_hashes", {})
        mismatches = {
            name: {"expected": expected, "actual": sha256_file(root / name)}
            for name, expected in expected_hashes.items()
            if (root / name).exists() and sha256_file(root / name) != expected
        }
        missing_hashes = [
            name
            for name in REQUIRED_RELEASE_FILES
            if name != "manifest.json" and name not in expected_hashes
        ]
        result["hashes_match"] = not mismatches and not missing_hashes
        result["hash_mismatches"] = mismatches
        result["missing_hashes"] = missing_hashes
        result["manifest"] = manifest
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        result["error"] = f"Không đọc được manifest: {exc}"
    return result
