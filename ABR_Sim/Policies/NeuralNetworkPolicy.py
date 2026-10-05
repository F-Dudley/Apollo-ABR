import os
import zmq
import msgpack
from enum import StrEnum
from typing import Any, Callable, TypedDict
from uuid import uuid4
from abc import ABC, abstractmethod

from ..Core.Interfaces import ABRPolicy
from ..Core.Types import ScenarioConfig, Action, BitrateLadder, SimulatorState


class NeuralPolicyMode(StrEnum):
    WORKER = "worker"
    PROXY = "proxy"


class NeuralPolicyRequest(TypedDict):
    request_id: str
    policy_id: str
    data: dict[str, Any]


class NeuralPolicyResponse(TypedDict):
    request_id: str
    policy_id: str
    selected_index: int
    error: str | None


class NeuralNetworkPolicy(ABC, ABRPolicy):

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        seed: int | None = None,
        mode: NeuralPolicyMode = NeuralPolicyMode.PROXY,
        endpoint: str = "tcp://127.0.0.1:6888",
    ):
        super().__init__(scenario_config=scenario_config, seed=seed)
        self.class_mode = mode

        # Worker Specifics
        self._socket: zmq.Socket = None
        self._context: zmq.Context = None
        self._endpoint = endpoint

        # Proxy Specifics
        self._model: Any = None

        match self.class_mode:
            case NeuralPolicyMode.WORKER:
                self._initialized = self._initialize_worker()

            case NeuralPolicyMode.PROXY:
                self._initialized = self._initialize_proxy()

            case _:
                raise ValueError(
                    f"Invalid mode {self.class_mode}. Must be one of {list(NeuralPolicyInit)}"
                )

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _initialize_worker(self) -> bool:
        self._model = self.load_model()

        return True

    def _initialize_proxy(self) -> bool:
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.DEALER)

        identity = f"NeuralNetworkPolicy-{os.getpid()}-{uuid4()}".encode("utf-8")
        self._socket.setsockopt(zmq.IDENTITY, identity)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

        return True

    def _is_proxy(self) -> bool:
        assert (
            self.class_mode == NeuralPolicyMode.PROXY
        ), f"Expected mode {NeuralPolicyMode.PROXY}, but got {self.class_mode}"

    def _is_worker(self) -> bool:
        assert (
            self.class_mode == NeuralPolicyMode.WORKER
        ), f"Expected mode {NeuralPolicyMode.WORKER}, but got {self.class_mode}"

    def is_initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    def load_model(self) -> Any: ...

    @abstractmethod
    def build_payload(
        self,
        state_t: SimulatorState,
        ladder: BitrateLadder,
        segment_number: int,
        max_segment_number: int,
        segment_lookup: Callable[[int], BitrateLadder],
    ) -> NeuralPolicyRequest: ...

    @abstractmethod
    def run_inference(self, payload: NeuralPolicyRequest) -> int: ...

    def select_action(
        self,
        state_t: SimulatorState,
        ladder: BitrateLadder,
        *,
        segment_number: int,
        max_segment_number: int,
        segment_lookup: Callable[[int], BitrateLadder],
    ) -> Action:
        self._is_proxy()

        payload = self.build_payload(
            state_t, ladder, segment_number, max_segment_number, segment_lookup
        )
        request_id = payload["request_id"]

        response = self._request(payload)

        if response["request_id"] != request_id:
            raise RuntimeError(
                f"Received response with mismatched request_id. Expected {request_id}, but got {response['request_id']}"
            )

        selected_index = response.get("selected_index")
        if selected_index is None:
            raise RuntimeError(
                f"Received response without 'selected_index' field. Response: {response}"
            )
        assert isinstance(
            selected_index, int
        ), f"Expected selected_index to be int, but got {type(selected_index)} - Response: {response}"

        entry = ladder.get_entry(selected_index)

        return Action(
            bitrate_index=selected_index,
            bitrate_kbps=entry["bitrate_kbps"],
            resolution_width=entry["resolution_width"],
            resolution_height=entry["resolution_height"],
            vmaf=entry["vmaf"],
            segment_size_bytes=entry["segment_size_bytes"],
        )

    def infer_action(self, payload: NeuralPolicyRequest) -> NeuralPolicyResponse:
        self._is_worker()

        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call 'load_model' before running inference."
            )

        selected_index = self.run_inference(payload)

        return NeuralPolicyResponse(
            request_id=payload["request_id"],
            policy_id=payload["policy_id"],
            selected_index=selected_index,
        )

    def _generate_neural_request_id(self) -> str:
        return uuid4().hex

    # Socket Based Methods
    def _request(self, payload: NeuralPolicyRequest) -> Action:
        self._is_proxy()

        packed_payload = msgpack.packb(payload, use_bin_type=True)
        self._socket.send(packed_payload)

        # Wait for response
        response = msgpack.unpackb(self._socket.recv(), raw=False)

        if response["request_id"] != payload["request_id"]:
            raise RuntimeError(
                f"Received response with mismatched request_id. Expected {payload['request_id']}, but got {response['request_id']}"
            )

        if response.get("error") is not None:
            raise RuntimeError(f"Error from worker: {response['error']}")

        return response
