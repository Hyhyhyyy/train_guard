"""Optional dependency helpers — never auto-install."""

from __future__ import annotations

from typing import Any, Optional


def try_import_yaml() -> Any:
    """Return PyYAML module or None."""
    try:
        import yaml  # type: ignore

        return yaml
    except ImportError:
        return None


def try_import_pil() -> Any:
    """Return PIL.Image or None."""
    try:
        from PIL import Image  # type: ignore

        return Image
    except ImportError:
        return None


def try_import_psutil() -> Any:
    """Return psutil module or None."""
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError:
        return None


def try_import_torch() -> Any:
    """Return torch module or None."""
    try:
        import torch  # type: ignore

        return torch
    except ImportError:
        return None


def package_version(name: str) -> Optional[str]:
    """Installed distribution version or None."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            aliases = {
                "llamafactory": ["llamafactory", "llama-factory", "LLaMA-Factory"],
                "pillow": ["Pillow", "pillow"],
                "pyyaml": ["PyYAML", "pyyaml"],
            }
            for alt in aliases.get(name.lower(), [name]):
                try:
                    return version(alt)
                except PackageNotFoundError:
                    continue
            return None
    except Exception:  # noqa: BLE001
        return None
