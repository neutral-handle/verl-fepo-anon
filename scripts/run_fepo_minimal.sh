#!/usr/bin/env bash
set -euo pipefail

# Anonymous minimal FEPO launcher.
# Run this script from the upstream verl repo root after applying patch.

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
TRAIN_FILES="${TRAIN_FILES:-/path/to/train.parquet}"
AIME24_VAL="${AIME24_VAL:-/path/to/aime2024_test.parquet}"
AIME25_VAL="${AIME25_VAL:-/path/to/aime2025_test.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/fepo_minimal}"

PROJECT_NAME="${PROJECT_NAME:-verl-fepo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fepo_v2_minimal}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

mkdir -p "${OUTPUT_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  +algorithm.fepo.enable=true \
  +algorithm.fepo.variant=lowtail_adv \
  +algorithm.fepo.high_entropy_select_mode=top_ratio \
  +algorithm.fepo.high_entropy_top_ratio=0.1 \
  +algorithm.fepo.alpha=0.2 \
  +algorithm.fepo.beta=0.2 \
  +algorithm.fepo.low_tail_neg_adv_penalty=0.1 \
  +algorithm.fepo.high_head_penalty=0.0 \
  +algorithm.fepo.suffix_mode=sentence \
  +algorithm.fepo.f_sentence_stop=simple \
  +algorithm.fepo.sentence_min_suffix_tokens=5 \
  +algorithm.fepo.sentence_only_high_entropy=true \
  +algorithm.fepo.sentence_high_entropy_ratio=0.05 \
  +algorithm.fepo.sentence_max_scan_tokens=64 \
  +algorithm.fepo.sentence_num_threads=16 \
  +algorithm.fepo.suffix_len_debias_enable=true \
  +algorithm.fepo.suffix_len_bin_width=3 \
  +algorithm.fepo.suffix_len_min_count=100 \
  +algorithm.fepo.suffix_len_z_clip=3.0 \
  +algorithm.fepo.entropy_top_p=1.0 \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="['${AIME25_VAL}','${AIME24_VAL}']" \
  data.train_batch_size=128 \
  data.val_batch_size=128 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  +trainer.fepo_dump_freq=50 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_epochs=5 \
  "$@"
