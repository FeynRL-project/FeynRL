from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trajectory:
    """One environment step: modality input → model action → reward signal.

    Attributes
    ----------
    input:
        Raw modality-specific input.  For text this is a ``list[int]`` of
        prompt token ids; for audio it is a ``np.ndarray`` waveform; for
        vision it is an image tensor.
    action:
        Generated response as token ids, shape ``[R]``.
    rendered_output:
        Reward-function-visible form of the action.  For text output this is
        a decoded string; for audio output it could be a waveform array.
    reward:
        Per-action-token reward tensor, shape ``[R]``.
    training_signal:
        Dict consumed by ``ReplayBuffer.add_batch_seqs`` and the trainer.
        Must contain at minimum: ``input_ids``, ``pred_rewards``,
        ``pred_zscores``, ``pred_masks``, ``pred_dones``,
        ``pred_old_logprobs``, ``policy_version``, ``response_len``.
        Encoder-decoder models additionally include an ``encoder_inputs`` key.
    metadata:
        Arbitrary pass-through data (solution strings, sample ids, …).
    """

    input: Any
    action: torch.Tensor
    rendered_output: Any
    reward: torch.Tensor
    training_signal: dict
    metadata: dict = field(default_factory=dict)


class ModalityAdapter(ABC):
    """Interface between modality-specific I/O and the generic RL loop.

    One adapter instance lives inside each rollout engine and trainer.
    Implement all five abstract methods to support a new modality.

    The adapter is the *only* place that knows about modality-specific
    encoding.  Everything above it (replay buffer, algorithms) operates on
    ``Trajectory`` or its ``training_signal`` dict and never inspects the
    raw ``input``.
    """

    # ------------------------------------------------------------------
    # Data feed layer
    # ------------------------------------------------------------------

    @abstractmethod
    def load_sample(self, raw: dict) -> tuple[Any, dict]:
        """Convert a raw dataset row into ``(input, metadata)``.

        Parameters
        ----------
        raw:
            A single row returned by the data feed (e.g. a ``PromptsFeed``
            ``__getitem__`` dict).

        Returns
        -------
        inp:
            Modality-specific input passed downstream to
            ``build_rollout_request`` and ``build_training_signal``.
        metadata:
            Auxiliary data (solution strings, sample ids, …) that travels
            alongside ``inp`` through the rollout pipeline.
        """

    # ------------------------------------------------------------------
    # Rollout layer
    # ------------------------------------------------------------------

    @abstractmethod
    def build_rollout_request(self, inp: Any, metadata: dict) -> dict:
        """Build a vLLM-compatible request dict from ``inp`` and ``metadata``.

        For text this reconstructs the ``{"prompt_token_ids": [...], ...}``
        dict that ``VLLMRolloutEngine.generate`` currently receives.
        For audio this adds a ``"multi_modal_data": {"audio": [...]}`` key.
        """

    @abstractmethod
    def parse_rollout_output(self, vllm_out: dict, request: dict) -> torch.Tensor:
        """Extract action token ids ``[R]`` from a completed vLLM output dict.

        ``vllm_out["input_ids"]`` is the full prompt + response sequence.
        The prompt length can be recovered from ``request``.
        """

    @abstractmethod
    def render_output(self, inp: Any, action: torch.Tensor, metadata: dict) -> Any:
        """Produce the reward-function-visible form of the action.

        The return value is passed to ``prompt_data`` / ``response_data`` in
        the reward function.  For text output this is a decoded string.
        """

    # ------------------------------------------------------------------
    # Training layer
    # ------------------------------------------------------------------

    @abstractmethod
    def build_training_signal(
        self,
        inp: Any,
        action: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        metadata: dict,
        policy_version: int = 0,
    ) -> dict:
        """Assemble the training-signal dict consumed by ``ReplayBuffer``.

        Called *after* reward normalisation; ``rewards`` already contains
        z-scored values when the engine uses group normalisation.

        The returned dict must contain:
            ``input_ids``          [T]  full prompt + response token ids
            ``pred_rewards``       [T]  per-token rewards (pred-aligned)
            ``pred_zscores``       [T]  z-scored rewards (pred-aligned)
            ``pred_masks``         [T]  1 at valid action positions, 0 elsewhere
            ``pred_dones``         [T]  1 at the final action token, 0 elsewhere
            ``pred_old_logprobs``  [T]  log-probabilities from the behaviour policy
            ``policy_version``     int  version of the policy that generated this
            ``response_len``       int  number of action tokens

        Encoder-decoder models additionally return:
            ``encoder_inputs``  dict | None  e.g. ``{"audio_values": Tensor, ...}``
        """
