import pytest

from verl.trainer.ppo.l2c_mode import L2CRewardScope, resolve_l2c_mode


def test_canonical_single_round_same_instance():
    mode = resolve_l2c_mode(
        {
            "l2c_reward_scope": "same_instance",
            "l2c_num_coaching_rounds": 1,
            "iter_compact_steps": 99,
        },
        "l2l",
    )

    assert mode.reward_scope is L2CRewardScope.SAME_INSTANCE
    assert mode.coaching_rounds == 1
    assert mode.actor_attempts == 2
    assert not mode.uses_legacy_iterations


def test_canonical_paper_k10_is_nine_coaching_rounds():
    mode = resolve_l2c_mode(
        {
            "l2c_reward_scope": "same_instance",
            "l2c_num_coaching_rounds": 9,
        },
        "l2l_eval",
    )

    assert mode.coaching_rounds == 9
    assert mode.actor_attempts == 10


def test_cross_instance_mode():
    mode = resolve_l2c_mode(
        {
            "l2c_reward_scope": "cross_instance",
            "l2c_num_coaching_rounds": 1,
        },
        "l2l",
    )
    assert mode.reward_scope is L2CRewardScope.CROSS_INSTANCE


def test_cross_instance_rejects_iterative_mode():
    with pytest.raises(ValueError, match="single-round"):
        resolve_l2c_mode(
            {
                "l2c_reward_scope": "cross_instance",
                "l2c_num_coaching_rounds": 2,
            },
            "l2l",
        )

def test_unknown_reward_scope_is_rejected():
    with pytest.raises(ValueError, match="Unknown trainer.l2c_reward_scope"):
        resolve_l2c_mode(
            {
                "l2c_reward_scope": "per_source_probe",
                "l2c_num_coaching_rounds": 1,
            },
            "l2l",
        )
