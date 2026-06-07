# PANDA: Proxy-Anchor Deepfake Attribution

This repository hosts the official source code and data protocols for our EUSIPCO 2026 submission:
> Cristian-Teodor Neamtu, Serban Mihalache, Stefan Smeu, Dan Oneata, Horia Cucu, Dragos Burileanu, "Anchoring the Unknown: Open-Set Model Attribution via Proxy-Anchor Learning", accepted at EUSIPCO 2026, Bruges, Belgium.


## Overview

We address **open-set TTS source tracing**: given a speech sample, identify which TTS architecture generated it, and detect samples from architectures unseen during training (OOD detection). The system trains a proxy-anchor metric learning network on top of pre-extracted SSL embeddings from the [MLAAD v9](https://huggingface.co/datasets/mueller91/MLAAD) dataset (140 TTS systems, 51 languages). OOD detection is calibrated on a held-out set of unseen systems targeting 95% recall.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download MLAAD v9 and extract SSL embeddings using `src/extract_ssl_embeddings.py`. The script supports any HuggingFace SSL model via `--model_name`.

```bash
# Wav2Vec2-BERT (used in the paper)
python src/extract_ssl_embeddings.py \
    --model_name facebook/w2v-bert-2.0 \
    --input_dir /path/to/MLAAD_v9/fake \
    --output_dir data/w2vbert_embeddings/layer4 \
    --layer 4

# WavLM-Large or HuBERT-Large (ablation)
python src/extract_ssl_embeddings.py \
    --model_name microsoft/wavlm-large \
    --input_dir /path/to/MLAAD_v9/fake \
    --output_dir data/wavlm_embeddings/layer4 \
    --layer 4
```

Input structure: `{input_dir}/{language}/{model_name}/{audio}.wav`
Output structure: `{output_dir}/{language}/{model_name}/{audio}.pt`

## Reproducing the Paper Experiments

The `metadata/` directory contains the exact model-level and sample-level splits used in the paper. Pass `--use_existing_split` to reproduce identical results.

**Experiment 1 — Non-merged (120 ID classes):**

```bash
python src/train.py \
    --use_existing_split \
    --embeddings_dir data/w2vbert_embeddings/layer4 \
    --metadata_dir metadata/non_merged \
    --checkpoint_dir checkpoints/non_merged

python src/evaluate.py \
    --embeddings_dir data/w2vbert_embeddings/layer4 \
    --metadata_dir metadata/non_merged \
    --checkpoint_dir checkpoints/non_merged \
    --output_dir results/non_merged/proxy_anchor
```

**Experiment 2 — Merged architectures (110 ID classes):**

```bash
python src/train.py \
    --use_existing_split \
    --embeddings_dir data/w2vbert_embeddings/layer4 \
    --merge_map configs/merge_map.json \
    --metadata_dir metadata/merged \
    --checkpoint_dir checkpoints/merged

python src/evaluate.py \
    --embeddings_dir data/w2vbert_embeddings/layer4 \
    --metadata_dir metadata/merged \
    --checkpoint_dir checkpoints/merged \
    --output_dir results/merged/proxy_anchor \
    --merge_map metadata/merged/merge_map.json
```

**Baselines (Logistic Regression and k-NN):**

```bash
python src/evaluate_baselines.py --metadata_dir metadata/non_merged --output_dir results/non_merged/baselines
python src/evaluate_baselines.py --metadata_dir metadata/merged     --output_dir results/merged/baselines
```

## Architecture Merging

`configs/merge_map.json` groups model variants sharing the same underlying architecture (e.g., Llasa-1B/3B/8B → "Llasa"). When `--merge_map` is provided, the train/test split operates at the architecture level so no architecture spans the ID and OOD sets. Models not listed in the map keep their original name.

## Acknowledgements

This work was in part supported by a grant of the Ministry of Research, Innovation and Digitization, CCCDI - UEFISCDI, project number PN-IV-P7-7.1-PTE-2024-0600, within PNCDI IV. This work was also partially supported by the European Union – NextGenerationEU, through the National Recovery and Resilience Plan (PNRR), Component 9, Investment 4, under project SENSE, THINK @ POLITEHNICA BUCUREȘTI (SENTHIPoli) No. RUE 6.PI/I4/C9. The content of this material does not necessarily represent the official position of the European Union or the Government of Romania.
