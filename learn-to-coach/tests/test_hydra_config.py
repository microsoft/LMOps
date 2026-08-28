from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException


CONFIG_DIR = str(
    Path(__file__).resolve().parents[1] / "verl" / "verl" / "trainer" / "config"
)


def compose_config(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="ppo_trainer", overrides=overrides)


def test_eval_only_fields_use_hydra_add_syntax():
    config = compose_config(
        [
            "trainer.stage=l2l_eval",
            "trainer.setting=math",
            "trainer.l2c_reward_scope=same_instance",
            "trainer.l2c_num_coaching_rounds=1",
            "actor_rollout_ref.rollout.n=16",
            "++actor_rollout_ref.rollout.seed=0",
            "++trainer.force_global_step=100",
        ]
    )
    assert config.actor_rollout_ref.rollout.seed == 0
    assert config.trainer.force_global_step == 100


def test_cross_math_batch_contains_sources_and_probes():
    config = compose_config(
        [
            "trainer.stage=l2l",
            "trainer.setting=math",
            "trainer.l2c_reward_scope=cross_instance",
            "trainer.l2c_num_coaching_rounds=1",
            "trainer.exp_learner_batch_size=64",
            "trainer.probe_size=8",
            "data.train_batch_size=72",
            "actor_rollout_ref.rollout.n=8",
        ]
    )
    assert config.data.train_batch_size == 72
    assert config.trainer.exp_learner_batch_size + config.trainer.probe_size == 72


def test_rollout_seed_is_not_in_the_base_schema():
    try:
        compose_config(["actor_rollout_ref.rollout.seed=0"])
    except ConfigCompositionException:
        return
    raise AssertionError("plain rollout.seed override unexpectedly succeeded")


if __name__ == "__main__":
    test_eval_only_fields_use_hydra_add_syntax()
    test_cross_math_batch_contains_sources_and_probes()
    test_rollout_seed_is_not_in_the_base_schema()
    print("Hydra configuration tests passed.")
