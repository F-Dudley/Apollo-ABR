import inspect
from typing import Callable

from ..Core.Interfaces import ABRPolicy


class PolicyRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, policy_name: str, policy_class: type):
        cls._registry[policy_name.lower()] = policy_class

    @classmethod
    def get_policy_class(cls, policy_name: str) -> type:
        policy_name = policy_name.lower()

        if policy_name not in cls._registry:
            raise ValueError(f"Policy '{policy_name}' is not registered.")
        return cls._registry[policy_name]

    @classmethod
    def available_policies(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def create_policy(cls, policy_name: str, **kwargs) -> ABRPolicy:
        policy_class = cls.get_policy_class(policy_name)

        signature = inspect.signature(policy_class.__init__)
        accepted = set(signature.parameters.keys()) - {"self"}

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        return policy_class(**filtered_kwargs)


def ABRPolicyClass(
    arg: type[ABRPolicy] | str | None = None,
    *,
    name: str | None = None,
) -> Callable[[type[ABRPolicy]], type[ABRPolicy]] | type[ABRPolicy]:

    def decorator(policy_class: type[ABRPolicy]) -> type[ABRPolicy]:
        policy_name = name

        if policy_name is None and isinstance(arg, str):
            policy_name = arg

        if policy_name is None:
            policy_name = policy_class.__name__

        policy_class.name = policy_name
        PolicyRegistry.register(policy_name, policy_class)

        return policy_class

    # Supports @ABRPolicyClass without brackets.
    if isinstance(arg, type):
        return decorator(arg)

    # Supports @ABRPolicyClass("random") and @ABRPolicyClass(name="random")
    return decorator


# Import all policy classes to ensure they are registered.
from .RandomPolicy import RandomPolicy
from .RandomWalkPolicy import RandomWalkPolicy
from .ThroughputPolicy import ThroughputPolicy
from .BOLAPolicy import BOLAPolicy
from .WISHPolicy import WISHPolicy

__all__ = [
    "RandomPolicy",
    "RandomWalkPolicy",
    "ThroughputPolicy",
    "BOLAPolicy",
    "WISHPolicy",
    "HybridPolicy",
]
