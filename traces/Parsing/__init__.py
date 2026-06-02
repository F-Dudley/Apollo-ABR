from dataclasses import dataclass
import hashlib
from typing import Callable
from pathlib import Path

from .BaseParser import TraceFileParser, TraceSummary


class ParserRegistry:
    parsers = {}
    hash_names = set()

    @classmethod
    def register_parser(cls, sources: list[str], parser_cls: type[TraceFileParser]):
        for source in sources:
            source = source.lower()
            cls.parsers[source] = parser_cls()

    @classmethod
    def get_parser(cls, source_name: str) -> TraceFileParser | None:
        name = source_name.lower()
        return cls.parsers.get(name)

    @classmethod
    def create_hash_name(cls, name: str, digest_size: int = 32) -> str:
        h = hashlib.blake2b(digest_size=digest_size)
        h.update(name.encode("utf-8"))
        hash_name = h.hexdigest()

        if hash_name in cls.hash_names:
            raise ValueError(
                f"Hash name collision for '{name}' with hash '{hash_name}'"
            )

        cls.hash_names.add(hash_name)
        return hash_name


def TraceParser(
    arg: type[TraceFileParser] | str | None = None,
    *,
    name: str | None = None,
    accepted_sources: list[str] | None = None,
) -> Callable[[type[TraceFileParser]], type[TraceFileParser]] | type[TraceFileParser]:

    def decorator(policy_class: type[TraceFileParser]) -> type[TraceFileParser]:
        policy_name = name

        if policy_name is None and isinstance(arg, str):
            policy_name = arg

        if policy_name is None:
            policy_name = policy_class.__name__

        policy_class.name = policy_name
        policy_class.accepted_sources = accepted_sources or []
        ParserRegistry.register_parser(accepted_sources, policy_class)

        return policy_class

    if isinstance(arg, type):
        return decorator(arg)

    return decorator


from .UCCParser import UCCParser
from .MERINAParser import MERINAParser

__all__ = [
    "TraceParser",
    "ParserRegistry",
    "UCCParser",
    "MERINAParser",
]
