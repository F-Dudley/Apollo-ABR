import inspect
from typing import Callable
import zmq
import msgpack
from threading import Thread, Event

from ..Core.Interfaces import ABRPolicy
from .NeuralNetworkPolicy import (
    NeuralNetworkPolicy,
    NeuralPolicyRequest,
    NeuralPolicyResponse,
    NeuralPolicyMode,
)


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
    def is_registered(cls, policy_name: str) -> bool:
        return policy_name.lower() in cls._registry

    @classmethod
    def create_policy(cls, policy_name: str, **kwargs) -> ABRPolicy:
        policy_class = cls.get_policy_class(policy_name)

        signature = inspect.signature(policy_class.__init__)
        accepted = set(signature.parameters.keys()) - {"self"}

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        return policy_class(**filtered_kwargs)


class NeuralPolicyManager:

    def __init__(self, endpoint: str = "tcp://127.0.0.1:6888"):

        self.endpoint = endpoint

        self._context = zmq.Context.instance()

        self._policies: dict[str, NeuralNetworkPolicy] = {}

        self._thread = Thread(target=self._run, daemon=True)
        self._stop_event = Event()

    def run(self):
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _run(self):
        socket: zmq.Socket = self._context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        try:
            while not self._stop_event.is_set():
                events = poller.poll(100)  # Wait for 0.1 second for a message
                if not events:
                    continue

                data = socket.recv_multipart()
                if len(data) != 2:
                    # Invalid message format; expecting [identity, packed_request]
                    continue

                identify, packed_request = data

                request = msgpack.unpackb(packed_request, raw=False)

                response = self._handle_request(request)

                socket.send_multipart(
                    [identify, msgpack.packb(response, use_bin_type=True)]
                )
        finally:
            socket.unbind(self.endpoint)
            socket.close()

    def _handle_request(self, request: NeuralPolicyRequest) -> NeuralPolicyResponse:
        request_id = request["request_id"]
        policy_id = request["policy_id"]

        if policy_id not in self._policies:
            self._lazy_load_policy(policy_id)

        policy = self._policies[policy_id]

        try:
            return policy.infer_action(request)
        except Exception as e:
            return NeuralPolicyResponse(
                request_id=request_id,
                policy_id=policy_id,
                selected_index=None,
                error=str(e),
            )

    def _lazy_load_policy(self, policy_id: str):
        policy_class = PolicyRegistry.get_policy_class(policy_id)

        if not issubclass(policy_class, NeuralNetworkPolicy):
            raise TypeError(
                f"Policy '{policy_id}' is not a subclass of NeuralNetworkPolicy."
            )

        policy_instance: NeuralNetworkPolicy = policy_class(
            scenario_config=None,
            seed=0,
            mode=NeuralPolicyMode.WORKER,
            endpoint=self.endpoint,
        )

        if not policy_instance.initialized:
            policy_instance.load_model()

        self._policies[policy_id] = policy_instance


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
from .VMAF_BOLAPolicy import VMAF_BOLAPolicy
from .WISHPolicy import WISHPolicy

__all__ = [
    "RandomPolicy",
    "RandomWalkPolicy",
    "ThroughputPolicy",
    "BOLAPolicy",
    "VMAF_BOLAPolicy",
    "WISHPolicy",
    "NeuralNetworkPolicy",
    "NeuralPolicyMode",
    "NeuralPolicyRequest",
    "NeuralPolicyResponse",
]
