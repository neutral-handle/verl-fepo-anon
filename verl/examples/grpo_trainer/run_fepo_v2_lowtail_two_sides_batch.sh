#!/usr/bin/env bash
# FEPO v2 (lowtail_adv) concise launcher - batch-rank mode
# - Low-tail boost: m = 1 + alpha when adv>0, q <= beta
# - Low-tail weaken: m = max(0, 1 - low_tail_neg_adv_penalty) when adv<0, q <= beta (default 0 disables)
# - High-head penalty: m = 1 - high_head_penalty, q >= 1 - beta
# - Rank scope is batch-level (all valid points in current batch)

set -euo pipefail

VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${VERL_ROOT}"

# -----------------------------
# Core experiment identity
# -----------------------------
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fepo_v2_lowtail_fast_qwen3_4b_batchrank}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/fepo_v2_lowtail_batchrank}"
mkdir -p "${OUTPUT_DIR}"

# -----------------------------
# Data
# -----------------------------
TRAIN_FILES="${TRAIN_FILES:-/path/to/train.parquet}"
MATH500_VAL="${MATH500_VAL:-/path/to/math500_test.parquet}"
AIME24_VAL="${AIME24_VAL:-/path/to/aime2024_test.parquet}"
USE_MATH_VERIFY_VAL="${USE_MATH_VERIFY_VAL:-1}"
CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-${VERL_ROOT}/examples/grpo_trainer/math_verify_val_reward.py}"

CUSTOM_REWARD_ARGS=()
if [ "${USE_MATH_VERIFY_VAL}" = "1" ]; then
  CUSTOM_REWARD_ARGS+=("custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}")
  CUSTOM_REWARD_ARGS+=("custom_reward_function.name=compute_score")
fi

# -----------------------------
# Training scale
# -----------------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"

# -----------------------------
# PPO / rollout
# -----------------------------
ACTOR_LR="${ACTOR_LR:-1e-6}"
WARM_UP_RATIO="${WARM_UP_RATIO:-0.05}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ROLLOUT_LOGPROB_MB_PER_GPU="${ROLLOUT_LOGPROB_MB_PER_GPU:-16}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
REF_LOGPROB_MB_PER_GPU="${REF_LOGPROB_MB_PER_GPU:-4}"

# -----------------------------
# Validation generation
# -----------------------------
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"

# -----------------------------
# FEPO v2 (lowtail_adv)
# -----------------------------
FEPO_ENABLE="${FEPO_ENABLE:-true}"
FEPO_VARIANT="${FEPO_VARIANT:-lowtail_adv}"
FEPO_H_THRESHOLD="${FEPO_H_THRESHOLD:-2.0}"
FEPO_ALPHA="${FEPO_ALPHA:-0.2}"
FEPO_BETA="${FEPO_BETA:-0.2}"
FEPO_LOW_TAIL_NEG_ADV_PENALTY="${FEPO_LOW_TAIL_NEG_ADV_PENALTY:-0.0}"
FEPO_HIGH_HEAD_PENALTY="${FEPO_HIGH_HEAD_PENALTY:-0.2}"
FEPO_HIGH_HEAD_NEG_ADV_BOOST="${FEPO_HIGH_HEAD_NEG_ADV_BOOST:-0.0}"
FEPO_RANK_SCOPE="${FEPO_RANK_SCOPE:-batch}"                      # group / batch
FEPO_LOW_TAIL_POS_ADV_ONLY="${FEPO_LOW_TAIL_POS_ADV_ONLY:-false}"
FEPO_HIGH_HEAD_NEG_ADV_ONLY="${FEPO_HIGH_HEAD_NEG_ADV_ONLY:-false}"

# Suffix entropy mode
FEPO_SUFFIX_MODE="${FEPO_SUFFIX_MODE:-fixed_window}"                  # sentence / full / fixed_window
FEPO_F_SENTENCE_STOP="${FEPO_F_SENTENCE_STOP:-simple}"           # simple / pysbd
FEPO_SENTENCE_MIN_SUFFIX_TOKENS="${FEPO_SENTENCE_MIN_SUFFIX_TOKENS:-5}"
FEPO_SENTENCE_ONLY_HIGH_ENTROPY="${FEPO_SENTENCE_ONLY_HIGH_ENTROPY:-true}"
FEPO_SENTENCE_HIGH_ENTROPY_RATIO="${FEPO_SENTENCE_HIGH_ENTROPY_RATIO:-0.01}"   # per-response cap
FEPO_SENTENCE_MAX_SCAN_TOKENS="${FEPO_SENTENCE_MAX_SCAN_TOKENS:-256}"           # per-point scan cap
FEPO_SENTENCE_NUM_THREADS="${FEPO_SENTENCE_NUM_THREADS:-64}"
FEPO_FIXED_WINDOW_TOKENS="${FEPO_FIXED_WINDOW_TOKENS:-16}"       # used when FEPO_SUFFIX_MODE=fixed_window
FEPO_SUFFIX_LEN_DEBIAS_ENABLE="${FEPO_SUFFIX_LEN_DEBIAS_ENABLE:-false}"
FEPO_SUFFIX_LEN_BIN_WIDTH="${FEPO_SUFFIX_LEN_BIN_WIDTH:-2}"
FEPO_SUFFIX_LEN_MIN_COUNT="${FEPO_SUFFIX_LEN_MIN_COUNT:-100}"
FEPO_SUFFIX_LEN_Z_CLIP="${FEPO_SUFFIX_LEN_Z_CLIP:-3.0}"

# Entropy estimator in FEPO path
FEPO_ENTROPY_TOP_P="${FEPO_ENTROPY_TOP_P:-0.95}"                 # 1.0 => full-vocab entropy

# -----------------------------
# Logging / dump
# -----------------------------
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
FEPO_DUMP_FREQ="${FEPO_DUMP_FREQ:-50}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUTPUT_DIR}/validation_data}"
FEPO_DATA_DIR="${FEPO_DATA_DIR:-${OUTPUT_DIR}/fepo_data}"

DUMP_ARGS=()
if [ -n "${ROLLOUT_DATA_DIR}" ]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
  DUMP_ARGS+=("trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}")
fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then
  mkdir -p "${VALIDATION_DATA_DIR}"
  DUMP_ARGS+=("trainer.validation_data_dir=${VALIDATION_DATA_DIR}")
fi
if [ -n "${FEPO_DATA_DIR}" ]; then
  mkdir -p "${FEPO_DATA_DIR}"
  DUMP_ARGS+=("+trainer.fepo_data_dir=${FEPO_DATA_DIR}")
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  +algorithm.fepo.enable="${FEPO_ENABLE}" \
  +algorithm.fepo.variant="${FEPO_VARIANT}" \
  +algorithm.fepo.h_threshold="${FEPO_H_THRESHOLD}" \
  +algorithm.fepo.alpha="${FEPO_ALPHA}" \
  +algorithm.fepo.beta="${FEPO_BETA}" \
  +algorithm.fepo.low_tail_neg_adv_penalty="${FEPO_LOW_TAIL_NEG_ADV_PENALTY}" \
  +algorithm.fepo.high_head_penalty="${FEPO_HIGH_HEAD_PENALTY}" \
  +algorithm.fepo.high_head_neg_adv_boost="${FEPO_HIGH_HEAD_NEG_ADV_BOOST}" \
  +algorithm.fepo.low_tail_pos_adv_only="${FEPO_LOW_TAIL_POS_ADV_ONLY}" \
  +algorithm.fepo.high_head_neg_adv_only="${FEPO_HIGH_HEAD_NEG_ADV_ONLY}" \
  +algorithm.fepo.rank_scope="${FEPO_RANK_SCOPE}" \
  +algorithm.fepo.suffix_mode="${FEPO_SUFFIX_MODE}" \
  +algorithm.fepo.f_sentence_stop="${FEPO_F_SENTENCE_STOP}" \
  +algorithm.fepo.sentence_min_suffix_tokens="${FEPO_SENTENCE_MIN_SUFFIX_TOKENS}" \
  +algorithm.fepo.sentence_only_high_entropy="${FEPO_SENTENCE_ONLY_HIGH_ENTROPY}" \
  +algorithm.fepo.sentence_high_entropy_ratio="${FEPO_SENTENCE_HIGH_ENTROPY_RATIO}" \
  +algorithm.fepo.sentence_max_scan_tokens="${FEPO_SENTENCE_MAX_SCAN_TOKENS}" \
  +algorithm.fepo.sentence_num_threads="${FEPO_SENTENCE_NUM_THREADS}" \
  +algorithm.fepo.fixed_window_tokens="${FEPO_FIXED_WINDOW_TOKENS}" \
  +algorithm.fepo.suffix_len_debias_enable="${FEPO_SUFFIX_LEN_DEBIAS_ENABLE}" \
  +algorithm.fepo.suffix_len_bin_width="${FEPO_SUFFIX_LEN_BIN_WIDTH}" \
  +algorithm.fepo.suffix_len_min_count="${FEPO_SUFFIX_LEN_MIN_COUNT}" \
  +algorithm.fepo.suffix_len_z_clip="${FEPO_SUFFIX_LEN_Z_CLIP}" \
  +algorithm.fepo.entropy_top_p="${FEPO_ENTROPY_TOP_P}" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="['${MATH500_VAL}','${AIME24_VAL}']" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${WARM_UP_RATIO}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  +trainer.fepo_dump_freq="${FEPO_DUMP_FREQ}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${CUSTOM_REWARD_ARGS[@]}" \
  "${DUMP_ARGS[@]}" \
  "$@"

