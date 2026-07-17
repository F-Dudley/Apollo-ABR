import os
from pathlib import Path
from typing import Any, Callable


class AssetRegistry:

    _assets: dict[str, Path] = {}
    _asset_index: dict[str, set[str]] = {}

    @classmethod
    def clear(cls):
        cls._assets.clear()
        cls._asset_index.clear()

    @classmethod
    def register(cls, asset_path: str | Path, *, recursive: bool = True):
        path = Path(asset_path).expanduser().resolve(strict=True)

        if not path.exists():
            raise FileNotFoundError(f"Asset path does not exist: {path}")

        if path.is_file():
            cls._register_file(path.name, path)

        elif path.is_dir():
            cls._register_directory(path, namespace=path.name, recursive=recursive)

        else:
            raise ValueError(f"Asset path is neither a file nor a directory: {path}")

    @classmethod
    def get(cls, key: str) -> Path:
        requested_key = cls._normalise_key(key)

        match = cls._assets.get(requested_key)

        if match is not None:
            return match

        basename = Path(requested_key).name

        matches = sorted(cls._asset_index.get(basename, set()))

        if len(matches) == 1:
            return cls._assets[matches[0]]

        if len(matches) > 1:
            choices = ", ".join(matches)

            raise KeyError(
                f"Multiple assets found for key '{requested_key}' (basename '{basename}'). "
                f"Please specify a more specific key. Choices: {choices}"
            )

        available = ", ".join(sorted(cls._assets))

        raise KeyError(
            f"No asset found for key '{requested_key}' (basename '{basename}'). "
            f"Available assets: {available}"
        )

    @classmethod
    def contain(cls, key: str) -> bool:
        try:
            cls.get(key)
        except KeyError:
            return False

        return True

    @classmethod
    def _register_file(cls, key: str, file_path: Path):
        key = cls._normalise_key(key)

        existing = cls._assets.get(key)

        if existing is not None:
            if existing == file_path:
                return  # Already registered, no action needed

            raise ValueError(
                f"Asset with key '{key}' is already registered at '{existing}'. Cannot register '{file_path}'."
            )

        cls._assets[key] = file_path
        cls._asset_index.setdefault(file_path.name, set()).add(key)

    @classmethod
    def _register_directory(cls, directory_path: Path, namespace: str, recursive: bool):
        iterator = directory_path.rglob("*") if recursive else directory_path.glob("*")
        iterator = filter(lambda p: p.is_file(), iterator)

        if iterator is None:
            raise ValueError(f"No valid files found in directory: {directory_path}")

        for file_path in sorted(iterator):
            relative_path = file_path.relative_to(directory_path).as_posix()

            key = f"{namespace}/{relative_path}"

            cls._register_file(key, file_path)

    @classmethod
    def _normalise_key(cls, key: str) -> str:
        key = key.replace("\\", "/")
        key = key.removeprefix("./")

        while "//" in key:
            key = key.replace("//", "/")

        return key.strip("/")
