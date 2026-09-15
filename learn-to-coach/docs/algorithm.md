# L2C algorithm and implementation map

## End-to-end dataflow

```text
source problem or environment seed
  -> frozen actor: initial solve trajectory
  -> LLM-as-a-Coach: n candidate knowledge snippets
       -> same-instance: each candidate guides a new solve on the source
       -> cross-instance: each candidate guides solves on disjoint probes
  -> verifier or environment reward
  -> normalize rewards across the n candidates from the same source
  -> GRPO update of the coach only
```

The actor is initialized before the coach and remains frozen. The historical
internal name `exp_learner` refers to the LLM-as-a-Coach.

## Protocol resolution

`verl/trainer/ppo/l2c_mode.py` resolves the two public configuration axes:

```yaml
trainer:
  stage: l2l
  setting: math
  l2c_reward_scope: same_instance
  l2c_num_coaching_rounds: 1
```

One coaching round contains Extract and Guided-solve. The initial solve is not
counted, hence the paper's attempt count is `K = rounds + 1`. The implementation
rejects cross-instance training with more than one round because candidate
carry and probe credit assignment are not defined for that setting.

## Same-instance training

For a source batch of size `B` and `n` candidates per source:

1. The actor generates one initial trajectory per source.
2. The coach generates `B * n` outputs in source-major order.
3. Each output guides a new actor solve on its own source.
4. Binary rewards are reshaped to `(B, n)` and normalized within each row.
5. Advantages are applied only to coach response tokens.

Iterative training repeats steps 2--5 inside one outer batch. Candidate slot
zero carries its knowledge and corresponding guided trajectory to the next
round. This deterministic choice has the same marginal distribution as a
uniform choice because samples from one prompt are exchangeable.

## Cross-instance training

The source and probe sets are disjoint. Math candidates share one probe pool
across the source batch. Text-game allocates an independent pool of
`trainer.probe_size` seeds per source group; only the `n` candidates from that
source share its probes. Each candidate receives its mean probe accuracy:

```text
pair_reward[candidate, probe] in {0, 1}
candidate_reward = mean(pair_reward[candidate, :])
```

Math flattens this grid in candidate-major, probe-minor order. Text-game
allocates `B * (1 + probe_size)` deterministic seeds per step, laid out as one
source followed by its probes for each group. Groups and steps have disjoint
seeds. Phase C runs `n` rounds of `B * probe_size` environments, using candidate
`source_idx * n + rollout_idx` on that source's probes. With `B=64`, `n=8`, and
`probe_size=8`, a step uses 64 source seeds, 512 distinct probe seeds, and 4096
probe rollouts. Transport-only padding handles batches not divisible by the
actor world size without changing logical source or probe counts.

## Evaluation

- Same-instance single-round evaluation reports initial, mean guided, and
  best-of-`n` accuracy.
- Iterative evaluation reports accuracy at every actor attempt from zero to
  `K - 1`, plus the oracle best across attempts.
- Cross-instance evaluation uses 64 sources and 250 disjoint probes by
  default. It reports source and probe baselines, cross-instance accuracy, and
  the candidate's own-source accuracy as a memorization diagnostic.

`trainer.phase_a_cache_path` can reuse frozen-actor trajectories across coach
checkpoints. Cache identity includes the actor model, tokenizer, prompt/chat
template, decoding configuration, dataset or environment ordering, and seed.

## Code ownership

| File | Responsibility |
|---|---|
| `verl/trainer/ppo/l2c_mode.py` | Canonical modes, paper-`K` mapping, invalid-combination checks |
| `verl/trainer/ppo/l2c.py` | Cross-instance rewards, source/probe evaluation, text-game padding/tokenization |
| `verl/trainer/ppo/l2c_train.py` | L2C training dispatch and coach GRPO updates |
| `verl/trainer/ppo/l2c_eval.py` | Same-instance, iterative, and Self-Refinement evaluation |
| `verl/trainer/ppo/ray_trainer.py` | Ray resource pools, workers, dataloaders, and checkpointing |
| `verl/trainer/config/ppo_trainer.yaml` | Public configuration defaults |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Actor generation and TextArena interaction |

## Invariants worth preserving

- Coach candidates stay source-major: `source_idx * n + rollout_idx`.
- Cross-math pairs stay candidate-major/probe-minor.
- Math verifier scores are explicitly converted from `+1/-1` to `0/1`.
- Text-game coach prompts use left padding.
- Source and probe identities are explicit; they are never inferred from an
  incidental row number after a shuffle.
- Checkpoint resume restores coach optimizer state.
