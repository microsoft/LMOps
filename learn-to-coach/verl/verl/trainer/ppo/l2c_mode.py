# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure configuration model for Learning to Coach (L2C)."""

from dataclasses import dataclass
from enum import Enum


class L2CRewardScope(str, Enum):
    """The instance set on which a candidate coaching note is rewarded."""

    SAME_INSTANCE = "same_instance"
    CROSS_INSTANCE = "cross_instance"


@dataclass(frozen=True)
class L2CMode:
    """Resolved L2C mode used by both training and evaluation.

    ``coaching_rounds`` counts Extract--Guided-solve rounds. The paper's
    number of actor attempts is therefore ``actor_attempts = rounds + 1``.
    """

    reward_scope: L2CRewardScope
    coaching_rounds: int
    actor_attempts: int
    uses_legacy_iterations: bool


def resolve_l2c_mode(trainer_config, stage: str) -> L2CMode:
    """Resolve canonical L2C axes while preserving old experiment drivers.

    New jobs should set ``l2c_num_coaching_rounds``. Historical jobs use
    ``iter_compact_steps`` with inconsistent semantics: training counts
    coaching rounds, while iterative evaluation counts total actor attempts.
    """

    scope_value = str(trainer_config.get("l2c_reward_scope", "same_instance")).lower()
    scope_aliases = {
        "same": L2CRewardScope.SAME_INSTANCE,
        "same_instance": L2CRewardScope.SAME_INSTANCE,
        "vanilla": L2CRewardScope.SAME_INSTANCE,
        "cross": L2CRewardScope.CROSS_INSTANCE,
        "cross_instance": L2CRewardScope.CROSS_INSTANCE,
        "meta": L2CRewardScope.CROSS_INSTANCE,
    }
    if scope_value not in scope_aliases:
        allowed = ", ".join(sorted(scope_aliases))
        raise ValueError(
            f"Unknown trainer.l2c_reward_scope={scope_value!r}; expected one of: {allowed}"
        )
    reward_scope = scope_aliases[scope_value]

    canonical_rounds = trainer_config.get("l2c_num_coaching_rounds", None)
    uses_legacy = canonical_rounds is None
    if canonical_rounds is not None:
        coaching_rounds = int(canonical_rounds)
    else:
        legacy_steps = int(trainer_config.get("iter_compact_steps", 1) or 1)
        if stage == "l2l_eval" and legacy_steps > 1:
            coaching_rounds = legacy_steps - 1
        else:
            coaching_rounds = legacy_steps

    if coaching_rounds < 1:
        raise ValueError(
            "trainer.l2c_num_coaching_rounds must be >= 1 "
            f"(got {coaching_rounds})"
        )
    if reward_scope is L2CRewardScope.CROSS_INSTANCE and coaching_rounds != 1:
        raise ValueError(
            "Cross-instance L2C currently implements the paper's single-round "
            "setting only: set trainer.l2c_num_coaching_rounds=1"
        )

    return L2CMode(
        reward_scope=reward_scope,
        coaching_rounds=coaching_rounds,
        actor_attempts=coaching_rounds + 1,
        uses_legacy_iterations=uses_legacy,
    )
