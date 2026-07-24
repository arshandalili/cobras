# COBRAS: Conditional Optimal Bridge for Riemannian Activation Steering

Implementation of the paper **"COBRAS: Conditional Optimal Bridge for Riemannian Activation Steering"**

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for running LLMs)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Install uv** (if not already installed): please follow instructions in https://docs.astral.sh/uv/getting-started/installation/

2. **Clone the repository**:
   ```bash
   git clone https://github.com/arshandalili/cobras.git
   cd cobras
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```
   This will automatically create a virtual environment and install all required packages.

4. **Set up API keys** (required for detoxification evaluation):

   For toxicity evaluation, you need a [Perspective API](https://perspectiveapi.com/) key:

   a. Create a `.env` file in the project root:
   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

   b. Export the environment file:
   ```bash
   export UV_ENV_FILE=.env
   ```

## Quick Start

Here's a minimal example to run COBRAS on TruthfulQA with Llama-3.1-8B:

```bash
# 1. Prepare data
uv run bash data/truthfulqa.sh

# 2. Generate responses with COBRAS
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
    model=Llama3.1-8B-Base \
    layer_idx=13 \
    steer=COBRAS \
    steer.T=0.5

# 3. Evaluate results
uv run python -u scripts/truthfulqa/truthfulqa_eval.py -m Llama3.1-8B-Base -l 13 -d
```

## Data Preparation

```bash
uv run bash data/ultrafeedback.sh
uv run bash data/truthfulqa.sh
uv run bash data/toxicity.sh
```

These scripts will:
- Download and format the each dataset
- Extract activations from the specified model at the specified layer

**Note**: For toxicity data preparation, you need Kaggle API credentials configured (`~/.kaggle/kaggle.json`).

## Running Experiments

### Helpfulness (Ultrafeedback)

**Generate responses**:
```bash
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=<MODEL> \
    layer_idx=<LAYER> \
    steer=<STEER> \
    steer.T=<T_VALUE>
```

**Evaluate responses**:
```bash
uv run python -u scripts/ultrafeedback/ultrafeedback_eval.py -m <MODEL> -l <LAYER> -d
```
The command will automatically evaluate all steering methods with the specific model and layer.

**Example - Compare COBRAS with baseline methods**:
```bash
# No steering (baseline)
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=NoSteer

# CAA
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=CAA steer.T=1.0

# COBRAS (our method)
uv run python -u scripts/ultrafeedback/ultrafeedback_generate.py \
    model=Llama3.1-8B-Base layer_idx=13 steer=COBRAS steer.T=0.5

# Evaluate
uv run python -u scripts/ultrafeedback/ultrafeedback_eval.py -m Llama3.1-8B-Base -l 13 -d
```

### Truthfulness (TruthfulQA)

**Generate responses**:
```bash
uv run python -u scripts/truthfulqa/truthfulqa_generate.py \
    model=<MODEL> \
    layer_idx=<LAYER> \
    steer=<STEER> \
    steer.T=<T_VALUE>
```

**Evaluate responses**:
```bash
uv run python -u scripts/truthfulqa/truthfulqa_eval.py -m <MODEL> -l <LAYER> -d
```


### Detoxification (RealToxicityPrompts)

**Generate responses**:
```bash
uv run python -u scripts/toxicity/detox_generate.py \
    model=<MODEL> \
    layer_idx=<LAYER> \
    steer=<STEER> \
    steer.T=<T_VALUE>
```

**Evaluate responses**:
```bash
uv run python -u scripts/toxicity/detox_eval.py -m <MODEL> -l <LAYER> -d
```

### Out-of-Distribution (MMLU, GSM8K, NQ, TriviaQA)

These tasks check that steering trained on an alignment dataset does not break general capabilities.

**MMLU**:
```bash
uv run python -u scripts/mmlu/mmlu_generate.py \
    model=<MODEL> layer_idx=<LAYER> steer=<STEER> steer.T=<T_VALUE>
uv run python -u scripts/mmlu/mmlu_eval.py -m <MODEL> -l <LAYER>
```

**GSM8K** (multi-GPU generation via `accelerate`, then merge shards):
```bash
uv run accelerate launch --num_processes <N> scripts/gsm8k/gsm8k_generate_fast.py \
    model=<MODEL> layer_idx=<LAYER> steer=<STEER> steer.T=<T_VALUE>
uv run python -u scripts/gsm8k/gsm8k_generate_merge.py \
    model=<MODEL> layer_idx=<LAYER> steer=<STEER> steer.T=<T_VALUE>
uv run python -u scripts/gsm8k/gsm8k_eval.py -m <MODEL> -l <LAYER>
```

**Natural Questions / TriviaQA** (closed-book multiple-choice transfer eval, following [ITI](https://arxiv.org/abs/2306.03341)):
```bash
uv run python -u scripts/nq/nq_generate.py \
    model=<MODEL> layer_idx=<LAYER> steer=<STEER> steer.T=<T_VALUE>
uv run python -u scripts/nq/nq_eval.py -m <MODEL> -l <LAYER>

uv run python -u scripts/triviaqa/triviaqa_generate.py \
    model=<MODEL> layer_idx=<LAYER> steer=<STEER> steer.T=<T_VALUE>
uv run python -u scripts/triviaqa/triviaqa_eval.py -m <MODEL> -l <LAYER>
```
Both use the ITI transfer splits ([`OamPatel/iti_nq_open_val`](https://huggingface.co/datasets/OamPatel/iti_nq_open_val),
[`OamPatel/iti_trivia_qa_val`](https://huggingface.co/datasets/OamPatel/iti_trivia_qa_val)), which pair each question with
its gold answers and a GPT-4-written plausible-but-false answer. The model is prompted closed-book with the TruthfulQA
instruction prompt and QA primer, each candidate answer is scored by its total log-likelihood, and a question counts as
correct when the top-scoring candidate is a gold answer (ITI: *"if the truthful answer ranks first, it contributes one
positive"*). NQ gold answers are the `answer` list; TriviaQA gold answers are the full alias list, matching ITI's loaders.

**Caveat on the released splits.** For 35% of TriviaQA questions (7% for NQ) the GPT-4 "false" answer is byte-identical to one of the gold answers, so those items cannot discriminate between methods and are scored correct by the rule above.
Absolute accuracies on TriviaQA are inflated accordingly; the datasets remain usable for the *relative* comparison between steering methods, which is what these OOD checks are for.

## Reproducing Paper Experiments

The `experiments/` directory bundles the full sweep used in the paper.

| Script | What it runs |
|--------|--------------|
| `experiments/runner.sh` | Top-level driver. Iterates over the four supported models, their recommended layer, and a per-model grid of `T` values, dispatching to the per-task scripts below. Tasks are toggled with env vars: `RUN_TQA`, `RUN_UF`, `RUN_TOXICITY`, `RUN_MMLU`, `RUN_GSM8K`, `RUN_NQ`, `RUN_TRIVIAQA`. |
| `experiments/experiments_truthfulqa.sh` | TruthfulQA: generates with every baseline + `COBRAS` across `REPEAT` seeds, then evaluates. |
| `experiments/experiments_ultrafeedback.sh` | UltraFeedback (helpfulness) sweep, same structure as above. |
| `experiments/experiments_toxicity.sh` | RealToxicityPrompts detoxification sweep. |
| `experiments/experiments_mmlu.sh` | MMLU OOD sweep. |
| `experiments/experiments_gsm8k.sh` | GSM8K OOD sweep (uses `accelerate` + merge step). |
| `experiments/experiments_nq.sh` | Natural Questions OOD sweep. |
| `experiments/experiments_triviaqa.sh` | TriviaQA OOD sweep. |

Example:
```bash
# In-distribution alignment sweeps (3 seeds each)
RUN_TQA=1 RUN_UF=1 RUN_TOXICITY=1 uv run bash experiments/runner.sh 3

# OOD capability checks (1 seed)
RUN_MMLU=1 RUN_GSM8K=1 RUN_NQ=1 RUN_TRIVIAQA=1 uv run bash experiments/runner.sh 1
```

## Adding a New Dataset

To run COBRAS and the baselines on a new dataset `<NAME>`:

1. **Data preparation.** Create `data/<NAME>/format_dataset.py` that downloads the source data and writes contrastive positive/negative JSONL files (with at minimum `idx`, prompt/question, and response/answer columns) under `data/<NAME>/texts/`. Follow [data/truthfulqa/format_dataset.py](data/truthfulqa/format_dataset.py) or [data/ultrafeedback/format_dataset.py](data/ultrafeedback/format_dataset.py) as templates.

2. **Activation extraction.** Add `data/<NAME>/extract_activations.py` that loads each JSONL, runs the model via `HuggingFaceLM`, and saves per-layer activation tensors and the matching `question_idx.pt`. The truthfulqa version is the simplest starting point.

3. **Driver script.** Add `data/<NAME>.sh` that calls the two scripts above for each supported model/layer.

4. **Hydra config.** Add `confs/<NAME>.yaml` mirroring [confs/truthfulqa.yaml](confs/truthfulqa.yaml); set `dataset: <NAME>` and the default model/layer.

5. **Generate + eval scripts.** Create `scripts/<NAME>/<NAME>_generate.py` (Hydra entrypoint reading `confs/<NAME>.yaml`) and `scripts/<NAME>/<NAME>_eval.py`. Reuse the loaders in `src/cobras/utils/data.py` — add a `load_<NAME>_data` helper if needed.

6. **Experiment runner.** Add `experiments/experiments_<NAME>.sh` (copy from `experiments_truthfulqa.sh`) and wire a `RUN_<NAME>` branch into `experiments/runner.sh`.

After these steps:
```bash
uv run bash data/<NAME>.sh
RUN_<NAME>=1 uv run bash experiments/runner.sh 3
```

## Supported Models and Methods

### Models

| Model Name | HuggingFace Path | Recommended Layer |
|------------|-----------------|-------------------|
| `Llama3.1-8B-Base` | `meta-llama/Llama-3.1-8B` | 13 |
| `Mistral-7B-Base` | `mistralai/Mistral-7B-v0.1` | 15 |
| `Falcon-7B-Base` | `tiiuae/falcon-7b` | 14 |
| `Qwen2.5-7B-Base` | `Qwen/Qwen2.5-7B` | 13 |

**Note:** The steering layer indices reported in our paper differ slightly from those used here, as this implementation starts counting from zero.

### Steering Methods

**Baselines:**

| Method | Description | References |
|--------|-------------|------------|
| `NoSteer` | Unmodified model forward pass; used as the reference baseline. | — |
| `RepE` | Adds a difference-of-means direction extracted via representation engineering to the hidden state. | [Paper](http://arxiv.org/abs/2310.01405), [Code](https://github.com/andyzoujm/representation-engineering) |
| `CAA` | Contrastive Activation Addition; shifts activations along a contrastive prompt-pair direction. | [Paper](https://aclanthology.org/2024.acl-long.828/), [Code](https://github.com/nrimsky/CAA) |
| `ITI` | Inference-Time Intervention; edits selected attention heads along truthful probe directions. | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html), [Code](https://github.com/likenneth/honest_llama) |
| `MiMiC` | Affine map that aligns negative-class activations to the positive-class distribution. | [Paper](https://openreview.net/forum?id=GwA4go0Mw4), [Code](https://github.com/shauli-ravfogel/affine-steering) |
| `LinAcT` | Learned linear activation transform fit on contrastive pairs. | [Paper](https://openreview.net/forum?id=l2zFn6TIQi), [Code](https://github.com/apple/ml-act) |
| `SphericalSteer` | Geodesic steering on the unit sphere via a vMF-style update. | — |
| `HPR` | Householder pseudo-rotation steering. | [Paper](https://aclanthology.org/2024.emnlp-main.761/), [Code](https://github.com/VinAIResearch/HPR) |
| `RE-Control` | DNN value-function-guided activation control. | [Paper](http://arxiv.org/abs/2406.05954), [Code](https://github.com/Lingkai-Kong/RE-Control) |
| `TruthFlow` | Flow-matching network that transports activations toward truthful targets. | [Paper](http://arxiv.org/abs/2502.04556), [Code](https://github.com/wwwhy725/TruthFlow) |
| `ODESteer` | Barrier function-guided ODE steering (prior method this repo is built on). | [Paper](https://arxiv.org/abs/2602.17560), [Code](https://github.com/ZhaoHongjue/odesteer) |
| `StepODESteer` | One-step ODESteer variant. | [Paper](https://arxiv.org/abs/2602.17560), [Code](https://github.com/ZhaoHongjue/odesteer) |

**Ours**:
- `COBRAS` - Conditional Optimal Bridge for Riemannian Activation Steering.

**Note:** For DNN-based steering methods (HPR, RE-Control, and TruthFlow), please refer to corresponding official repos for detailed implementation.


## Project Structure

```
cobras/
├── src/cobras/             # Main package
│   ├── lm/                 # Language model wrappers
│   ├── steer/              # Steering methods implementation
│   └── utils/              # Utilities (kernels, metrics, data loading)
├── scripts/                # Experiment scripts
│   ├── ultrafeedback/      # Ultrafeedback generation & evaluation
│   ├── truthfulqa/         # TruthfulQA generation & evaluation
│   ├── toxicity/           # Detoxification generation & evaluation
│   ├── mmlu/               # MMLU OOD generation & evaluation
│   ├── gsm8k/              # GSM8K OOD generation & evaluation
│   ├── nq/                 # Natural Questions OOD generation & evaluation
│   └── triviaqa/           # TriviaQA OOD generation & evaluation
├── data/                   # Data preparation scripts
│   ├── ultrafeedback/      # Ultrafeedback data preprocessing
│   ├── truthfulqa/         # TruthfulQA data processing
│   └── toxicity/           # Toxicity data processing
├── experiments/            # End-to-end sweep scripts
├── confs/                  # Hydra configuration files
├── results/                # Generated outputs (auto-created)
└── pyproject.toml          # Project dependencies
```

## Acknowledgements

This codebase is largely adapted from the official **ODESteer** implementation. The data pipelines, model wrappers, baseline integrations, and Hydra-based experiment scaffolding all originate from that repository; on top of it, COBRAS adds the Riemannian conditional optimal-bridge steering method and the out-of-distribution evaluation suite (MMLU, GSM8K, Natural Questions and TriviaQA generation, evaluation, and
`experiments/` sweep scripts). Many thanks to the ODESteer authors for releasing their code.

- Code: https://github.com/ZhaoHongjue/odesteer
- Paper: https://arxiv.org/abs/2602.17560

The Natural Questions and TriviaQA transfer evaluation follows **Inference-Time Intervention (ITI)** and uses the
adversarial-answer splits released by its authors.

- Code: https://github.com/likenneth/honest_llama
- Paper: https://arxiv.org/abs/2306.03341
