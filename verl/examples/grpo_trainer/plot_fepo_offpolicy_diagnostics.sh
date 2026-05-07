#!/usr/bin/env bash
# Plot FEPO off-policy diagnostics from W&B history.
# Produces:
#   1) importance-ratio distribution trend
#   2) suffix-rank Spearman trend
#
# Usage examples:
#   # Single run
#   RUN_PATH="myteam/verl/abc123" \
#   bash verl/examples/grpo_trainer/plot_fepo_offpolicy_diagnostics.sh
#
#   # Multi-run compare (same two figures, one line per method)
#   ENTITY="myteam" PROJECT="verl" \
#   RUNS="abc123 def456 ghi789" \
#   METHODS="FEPO GRPO GTPO" \
#   bash verl/examples/grpo_trainer/plot_fepo_offpolicy_diagnostics.sh

set -euo pipefail

VERL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${VERL_ROOT}"

# -----------------------------
# Input selection
# -----------------------------
RUN_PATH="${RUN_PATH:-}"                 # entity/project/run_id (single run mode)
ENTITY="${ENTITY:-}"                     # required in multi-run mode
PROJECT="${PROJECT:-}"                   # required in multi-run mode
RUNS="${RUNS:-}"                         # space-separated run ids (multi-run mode)
METHODS="${METHODS:-}"                   # space-separated method names (multi-run mode)

# -----------------------------
# Plot behavior
# -----------------------------
OUT_DIR="${OUT_DIR:-${VERL_ROOT}/plots/fepo_offpolicy_diag}"
MIN_STEP="${MIN_STEP:-0}"
MAX_STEP="${MAX_STEP:-0}"
SMOOTH="${SMOOTH:-1}"
X_LABEL="${X_LABEL:-step}"
X_TICK_MULTIPLE="${X_TICK_MULTIPLE:-100}"

# Metrics
METRIC_RATIO="fepo/offpolicy_ratio_p10,fepo/offpolicy_ratio_p50,fepo/offpolicy_ratio_p90,fepo/offpolicy_abs_log_ratio_mean"
METRIC_SPEARMAN="fepo/offpolicy_spearman_p10,fepo/offpolicy_spearman_p50,fepo/offpolicy_spearman_p90,fepo/offpolicy_spearman_mean"

mkdir -p "${OUT_DIR}"

PLOT_PY="${VERL_ROOT}/scripts/plot_wandb_run.py"
if [[ ! -f "${PLOT_PY}" ]]; then
  echo "error: plot script not found: ${PLOT_PY}" >&2
  exit 1
fi

echo "[plot] output dir: ${OUT_DIR}"

if [[ -n "${RUNS}" ]]; then
  # ---------- Multi-run mode ----------
  if [[ -z "${ENTITY}" || -z "${PROJECT}" ]]; then
    echo "error: multi-run mode requires ENTITY and PROJECT" >&2
    exit 1
  fi
  if [[ -z "${METHODS}" ]]; then
    echo "error: multi-run mode requires METHODS (space-separated)" >&2
    exit 1
  fi

  # shellcheck disable=SC2206
  RUN_ARR=(${RUNS})
  # shellcheck disable=SC2206
  METHOD_ARR=(${METHODS})
  if [[ ${#RUN_ARR[@]} -ne ${#METHOD_ARR[@]} ]]; then
    echo "error: RUNS count (${#RUN_ARR[@]}) != METHODS count (${#METHOD_ARR[@]})" >&2
    exit 1
  fi

  python "${PLOT_PY}" \
    --entity "${ENTITY}" \
    --project "${PROJECT}" \
    --runs "${RUN_ARR[@]}" \
    --methods "${METHOD_ARR[@]}" \
    --metrics "${METRIC_RATIO},${METRIC_SPEARMAN}" \
    --out-dir "${OUT_DIR}" \
    --min-step "${MIN_STEP}" \
    --max-step "${MAX_STEP}" \
    --smooth "${SMOOTH}" \
    --x-label "${X_LABEL}" \
    --x-tick-multiple "${X_TICK_MULTIPLE}"

else
  # ---------- Single-run mode ----------
  if [[ -z "${RUN_PATH}" ]]; then
    echo "error: single-run mode requires RUN_PATH=entity/project/run_id" >&2
    exit 1
  fi

  python "${PLOT_PY}" \
    --run-path "${RUN_PATH}" \
    --metrics "${METRIC_RATIO}" \
    --layout overlay \
    --output "${OUT_DIR}/offpolicy_ratio_trend.png" \
    --min-step "${MIN_STEP}" \
    --max-step "${MAX_STEP}" \
    --smooth "${SMOOTH}" \
    --x-label "${X_LABEL}" \
    --x-tick-multiple "${X_TICK_MULTIPLE}"

  python "${PLOT_PY}" \
    --run-path "${RUN_PATH}" \
    --metrics "${METRIC_SPEARMAN}" \
    --layout overlay \
    --output "${OUT_DIR}/offpolicy_spearman_trend.png" \
    --min-step "${MIN_STEP}" \
    --max-step "${MAX_STEP}" \
    --smooth "${SMOOTH}" \
    --x-label "${X_LABEL}" \
    --x-tick-multiple "${X_TICK_MULTIPLE}"
fi

echo "[done] FEPO off-policy diagnostics plots generated."

