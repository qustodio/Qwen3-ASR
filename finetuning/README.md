## Qustodio Qwen3-ASR

In Qustodio we've implemented a new script `serve_aligner.py` that first runs the Qwen3-ASR model and right after
the Aligner model is run so that timestamps are returned in the response.

In order to get as much performance as possible, both models need to run on GPU. Therefore, when running the script it's
recommended to add the following parameters:

* `--gpu-memory-utilization 0.5` so that half of the GPU is reserved for the ASR model and the rest can be used by the Aligner model.
* `--aligner-device cuda:0` to tell the Aligner model to use the GPU. This parameter can be replaced with the `QWEN_ASR_ALIGNER_DEVICE` environment variable.

The Aligner model can run in a batched mode (recommended). There are 2 environment variables that control batching on the Aligner model:

* `QWEN_ASR_ALIGN_BATCH_MAX`: If `>= 2` it merges up to that many concurrent calls to the model.
* `QWEN_ASR_ALIGN_BATCH_WAIT_MS`: That latency (in ms) is added per flush when the batch is not full.

### Building the Docker image

    docker buildx build --platform linux/amd64 \
        -f docker/Dockerfile-qwen3-asr-cu128 \
        -t "${ECR_URI}/${ECR_REPO}:20260505" --push --provenance=false \
        --build-arg BUNDLE_FLASH_ATTENTION=false .

Please note that:
* `--provenance=false` is set for compatibility with AWS ECR
* `--BUNDLE_FLASH_ATTENTION=false` is set because FlashAttention-2 dropped support for T4 GPUs (the one in g4dn.xlarge instance) so it can't be executed - FlashInfer is used instead. In case we upgrade to g5 instances, we might consider enabling this flag.


## Fine-tuning Qwen3-ASR

This script fine-tunes **Qwen3-ASR** using JSONL audio-text pairs. It supports multi-GPU training via `torchrun`.

### 1) Setup

First, please install the two Python packages `qwen-asr` and `datasets` using the command below.

```bash
pip install -U qwen-asr datasets
```

Then, to reduce GPU memory usage and speed up training, it is recommended to install FlashAttention 2.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

### 2) Input JSONL format

Prepare your training file as JSONL (one JSON per line). Each line must contain:

- `audio`: path to a WAV file
- `text`: transcript text (you can include a language prefix)

Example:
```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language English<asr_text>Another example."}
{"audio":"/data/wavs/utt0003.wav","text":"language English<asr_text>Fine-tuning data line."}
```

Language prefix recommendation:

- If you **have** language info, use:
  - `language English<asr_text>...`
  - `language Chinese<asr_text>...`
- If you **do not have** language info, use:
  - `language None<asr_text>...`

Note:
- If you set `language None`, the model will not learn language detection from that prefix.

### 3) Fine-tune (single GPU)

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

Checkpoints will be written to:
- `./qwen3-asr-finetuning-out/checkpoint-<global_step>`

### 4) Fine-tune (multi GPU with torchrun)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200
```

### 5) Resume training

Option A: explicitly set a checkpoint path:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume_from ./qwen3-asr-finetuning-out/checkpoint-200
```

Option B: automatically resume from the latest checkpoint under `output_dir`:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume 1
```

### 6) Quick inference test

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "qwen3-asr-finetuning-out/checkpoint-200",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
)

print(results[0].language)
print(results[0].text)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
TRAIN_FILE="./train.jsonl"
EVAL_FILE="./eval.jsonl"
OUTPUT_DIR="./qwen3-asr-finetuning-out"

torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --eval_file ${EVAL_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --log_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 5 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2
```