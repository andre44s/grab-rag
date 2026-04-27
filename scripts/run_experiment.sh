#!/bin/bash
set -e
cd "$(dirname "$0")/.."

RUN="conda run -n grab-rag python -m src.runner"

echo "phi x nq"
$RUN --models phi --datasets nq --policies all --conditions all --output results/phi-nq/
$RUN --models phi --datasets nq --policies CB --conditions Qclosed --output results/phi-nq/

echo "phi x hpqa"
$RUN --models phi --datasets hpqa --policies all --conditions all --output results/phi-hpqa/
$RUN --models phi --datasets hpqa --policies CB --conditions Qclosed --output results/phi-hpqa/

echo "llama x nq"
$RUN --models llama --datasets nq --policies all --conditions all --output results/llama-nq/
$RUN --models llama --datasets nq --policies CB --conditions Qclosed --output results/llama-nq/

echo "llama x hpqa"
$RUN --models llama --datasets hpqa --policies all --conditions all --output results/llama-hpqa/
$RUN --models llama --datasets hpqa --policies CB --conditions Qclosed --output results/llama-hpqa/

echo "qwen x nq"
$RUN --models qwen --datasets nq --policies all --conditions all --output results/qwen-nq/
$RUN --models qwen --datasets nq --policies CB --conditions Qclosed --output results/qwen-nq/

echo "qwen x hpqa"
$RUN --models qwen --datasets hpqa --policies all --conditions all --output results/qwen-hpqa/
$RUN --models qwen --datasets hpqa --policies CB --conditions Qclosed --output results/qwen-hpqa/

echo "seed43 llama x nq"
$RUN --models llama --datasets nq --policies all --conditions Q0 Q50 Q100 QC --seed 43 --output results/seed43-llama-nq/

echo "finished"
