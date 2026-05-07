#!/usr/bin/env bash
# FEPO training example (GRPO + token-level FEPO advantage shaping).
#
# All paths (model/data/output) must be provided via environment variables.
# No cluster- or user-specific absolute paths are hard-coded in this script.

set -euo pipefail

VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${VERL_ROOT}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"  # optional; inject externally


MODEL_PATH="${MODEL_PATH:-/path/to/model}"

PROJECT_NAME="${PROJECT_NAME:-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fepo_qwen3_4b_dapo_public_like}"

# Checkpoint / trainer outputs directory (trainer.default_local_dir)
# OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/verl_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/fepo_legacy_mc_bonus}"

mkdir -p "${OUTPUT_DIR}"
echo "Checkpoint / trainer output dir: ${OUTPUT_DIR}"

# Training data (parquet)
TRAIN_FILES="${TRAIN_FILES:-/path/to/train.parquet}"

# Validation: two datasets (OmegaConf list string)
MATH500_VAL="${MATH500_VAL:-/path/to/math500_test.parquet}"
AIME24_VAL="${AIME24_VAL:-/path/to/aime2024_test.parquet}"
USE_MATH_VERIFY_VAL="${USE_MATH_VERIFY_VAL:-1}"
CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-${VERL_ROOT}/examples/grpo_trainer/math_verify_val_reward.py}"

CUSTOM_REWARD_ARGS=()
if [ "${USE_MATH_VERIFY_VAL}" = "1" ]; then
  CUSTOM_REWARD_ARGS+=("custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH}")
  CUSTOM_REWARD_ARGS+=("custom_reward_function.name=compute_score")
  echo "Validation scorer routing enabled: ${CUSTOM_REWARD_FUNCTION_PATH}"
fi

# Actor LR warmup ratio over total steps (optim.lr_warmup_steps_ratio). Set 0 to disable.
WARM_UP_RATIO="${WARM_UP_RATIO:-0.05}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ROLLOUT_LOGPROB_MB_PER_GPU="${ROLLOUT_LOGPROB_MB_PER_GPU:-16}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
REF_LOGPROB_MB_PER_GPU="${REF_LOGPROB_MB_PER_GPU:-4}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUTPUT_DIR}/validation_data}"
FEPO_DATA_DIR="${FEPO_DATA_DIR:-${OUTPUT_DIR}/fepo_data}"

# FEPO configuration
FEPO_ENABLE="${FEPO_ENABLE:-true}"
FEPO_H_THRESHOLD="${FEPO_H_THRESHOLD:-2.0}"
FEPO_MAX_POINTS_PER_SEQ="${FEPO_MAX_POINTS_PER_SEQ:-0}"  # <=0: use dynamic budget (see FEPO_MAX_POINTS_RATIO)
FEPO_MAX_POINTS_RATIO="${FEPO_MAX_POINTS_RATIO:-0.01}"
FEPO_MC_M="${FEPO_MC_M:-1}"
FEPO_MC_TEMPERATURE="${FEPO_MC_TEMPERATURE:-1.0}"
FEPO_MC_TOP_P="${FEPO_MC_TOP_P:-0.95}"
FEPO_LOGPROBS_K="${FEPO_LOGPROBS_K:-20}"
FEPO_F_CONTINUATION_MODE="${FEPO_F_CONTINUATION_MODE:-first_sentence}"
FEPO_F_SENTENCE_MAX_NEW_TOKENS="${FEPO_F_SENTENCE_MAX_NEW_TOKENS:-128}"
FEPO_NORMALIZE_BY_CONTINUATION_LENGTH="${FEPO_NORMALIZE_BY_CONTINUATION_LENGTH:-true}"
FEPO_MC_MAX_NEW_TOKENS="${FEPO_MC_MAX_NEW_TOKENS:-128}"
FEPO_CANDIDATE_TOP_P="${FEPO_CANDIDATE_TOP_P:-0.95}"
FEPO_CANDIDATE_MAX_K="${FEPO_CANDIDATE_MAX_K:-5}"
FEPO_CANDIDATE_MIN_PROB="${FEPO_CANDIDATE_MIN_PROB:-0.05}"  # drop candidates with p_c < threshold (always keep chosen)
FEPO_MIN_CANDIDATES="${FEPO_MIN_CANDIDATES:-2}"
FEPO_PROBE_BATCH_CHUNK="${FEPO_PROBE_BATCH_CHUNK:-64}"  # candidate-probing batch chunk
FEPO_MC_BATCH_CHUNK="${FEPO_MC_BATCH_CHUNK:-64}"
FEPO_JOB_CONCURRENCY="${FEPO_JOB_CONCURRENCY:-16}"  # per-replica concurrent FEPO jobs
FEPO_GLOBAL_POOLING="${FEPO_GLOBAL_POOLING:-false}"  # enable global pooling across jobs
FEPO_F_BAR_MODE="${FEPO_F_BAR_MODE:-branching}"  # branching / prefix_minus_ht
FEPO_F_REAL_MODE="${FEPO_F_REAL_MODE:-teacher_forced_real_path}"  # chosen_branch_mc / teacher_forced_real_path
FEPO_DELTA_POS_THRESHOLD="${FEPO_DELTA_POS_THRESHOLD:-0.1}"
FEPO_DELTA_NEG_THRESHOLD="${FEPO_DELTA_NEG_THRESHOLD:-0.1}"
FEPO_BONUS_POS="${FEPO_BONUS_POS:-0.02}"
FEPO_BONUS_NEG="${FEPO_BONUS_NEG:-0.02}"
FEPO_DUMP_FREQ="${FEPO_DUMP_FREQ:-50}"  # dump FEPO JSONL every N steps

# Note: FEPO_MAX_POINTS_PER_SEQ is dynamic when <=0 (uses max_points_ratio).
# FEPO_MC_BATCH_CHUNK is fixed.

DUMP_ARGS=()
if [ -n "${ROLLOUT_DATA_DIR}" ]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
  DUMP_ARGS+=("trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}")
  echo "Rollout data dump enabled: ${ROLLOUT_DATA_DIR}"
fi
if [ -n "${VALIDATION_DATA_DIR}" ]; then
  mkdir -p "${VALIDATION_DATA_DIR}"
  DUMP_ARGS+=("trainer.validation_data_dir=${VALIDATION_DATA_DIR}")
  echo "Validation generations dump enabled: ${VALIDATION_DATA_DIR}"
fi
if [ -n "${FEPO_DATA_DIR}" ]; then
  mkdir -p "${FEPO_DATA_DIR}"
  DUMP_ARGS+=("+trainer.fepo_data_dir=${FEPO_DATA_DIR}")
  echo "FEPO sampled points dump enabled: ${FEPO_DATA_DIR}"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    +algorithm.fepo.enable="${FEPO_ENABLE}" \
    +algorithm.fepo.h_threshold="${FEPO_H_THRESHOLD}" \
    +algorithm.fepo.max_points_per_seq="${FEPO_MAX_POINTS_PER_SEQ}" \
    +algorithm.fepo.max_points_ratio="${FEPO_MAX_POINTS_RATIO}" \
    +algorithm.fepo.mc_m="${FEPO_MC_M}" \
    +algorithm.fepo.mc_temperature="${FEPO_MC_TEMPERATURE}" \
    +algorithm.fepo.mc_top_p="${FEPO_MC_TOP_P}" \
    +algorithm.fepo.logprobs_k="${FEPO_LOGPROBS_K}" \
    +algorithm.fepo.f_continuation_mode="${FEPO_F_CONTINUATION_MODE}" \
    +algorithm.fepo.f_sentence_max_new_tokens="${FEPO_F_SENTENCE_MAX_NEW_TOKENS}" \
    +algorithm.fepo.normalize_by_continuation_length="${FEPO_NORMALIZE_BY_CONTINUATION_LENGTH}" \
    +algorithm.fepo.mc_max_new_tokens="${FEPO_MC_MAX_NEW_TOKENS}" \
    +algorithm.fepo.candidate_top_p="${FEPO_CANDIDATE_TOP_P}" \
    +algorithm.fepo.candidate_max_k="${FEPO_CANDIDATE_MAX_K}" \
    +algorithm.fepo.candidate_min_prob="${FEPO_CANDIDATE_MIN_PROB}" \
    +algorithm.fepo.min_candidates="${FEPO_MIN_CANDIDATES}" \
    +algorithm.fepo.probe_batch_chunk="${FEPO_PROBE_BATCH_CHUNK}" \
    +algorithm.fepo.mc_batch_chunk="${FEPO_MC_BATCH_CHUNK}" \
    +algorithm.fepo.job_concurrency="${FEPO_JOB_CONCURRENCY}" \
    +algorithm.fepo.global_pooling="${FEPO_GLOBAL_POOLING}" \
    +algorithm.fepo.f_bar_mode="${FEPO_F_BAR_MODE}" \
    +algorithm.fepo.f_real_mode="${FEPO_F_REAL_MODE}" \
    +algorithm.fepo.delta_pos_threshold="${FEPO_DELTA_POS_THRESHOLD}" \
    +algorithm.fepo.delta_neg_threshold="${FEPO_DELTA_NEG_THRESHOLD}" \
    +algorithm.fepo.bonus_pos="${FEPO_BONUS_POS}" \
    +algorithm.fepo.bonus_neg="${FEPO_BONUS_NEG}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="['${MATH500_VAL}','${AIME24_VAL}']" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${WARM_UP_RATIO}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
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
