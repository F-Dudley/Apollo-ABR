import inspect

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


def ABRPolicyClass(cls, name: str | None = None):
    def decorator(cls):
        policy_name = name or cls.__name__

        cls.name = policy_name
        PolicyRegistry.register(policy_name, cls)
        return cls

    return decorator
