# PANDA: Proxy-Anchor Deepfake Attribution

This repository hosts the official source code and data protocols for our EUSIPCO 2026 paper:
> Cristian-Teodor Neamtu, Serban Mihalache, Stefan Smeu, Dan Oneata, Horia Cucu, Dragos Burileanu, "Anchoring the Unknown: Open-Set Model Attribution via Proxy-Anchor Learning", accepted at EUSIPCO 2026, Bruges, Belgium.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2606.10758)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)

## Abstract

The proliferation of text-to-speech (TTS) systems capable of generating realistic synthetic speech poses growing challenges for audio forensics. While binary deepfake detection has received considerable attention, source tracing (i.e., identifying which TTS system produced a given audio sample) remains underexplored, particularly in open-set scenarios where unknown systems may be encountered. We propose a metric learning framework based on the Proxy-Anchor loss function that operates on Wav2Vec2-BERT embeddings to learn a discriminative embedding space for TTS source attribution and out-of-distribution (OOD) detection of unseen systems. We evaluate it on the MLAAD v9 dataset spanning 140 TTS systems across 51 languages, and introduce an architecture merging strategy that groups TTS system versions into unified classes, reducing inter-class confusion. Our system achieves 99.76\% accuracy on 110 in-distribution classes and a False Positive Rate (FPR@95) as low as 2.04\% for OOD detection. Also, for a fair comparison against the current state of the art, we further evaluate it on the MLAAD v5 official dataset splits, improving the OOD accuracy by almost doubling it. These results demonstrate that Proxy-Anchor metric learning, combined with architecture-aware class design and post-hoc OOD scoring, provides an effective framework for forensic TTS source tracing in both closed-set and open-set settings.

## Citation

```bibtex
@inproceedings{neamtu2026panda,
  title     = {Anchoring the Unknown: Open-Set Model Attribution via Proxy-Anchor Learning},
  author    = {Neamtu, Cristian-Teodor and Mihalache, Serban and Smeu, Stefan and Oneata, Dan and Cucu, Horia and Burileanu, Dragos},
  booktitle = {Proceedings of the European Signal Processing Conference (EUSIPCO)},
  year      = {2026},
  address   = {Bruges, Belgium},
  note      = {To appear}
}
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download [MLAAD v9](https://huggingface.co/datasets/mueller91/MLAAD) and extract SSL embeddings using `src/extract_ssl_embeddings.py`. The script supports any HuggingFace SSL model via `--model_name`.

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

## Results

All results use **Wav2Vec2-BERT** (layer 4) embeddings and are reported on the held-out test set.
OOD detection thresholds are calibrated on a separate OOD calibration split at 95% TPR.

### Experiment 1 — Non-merged (120 ID classes)

| Method | Accuracy ↑ | AUROC ↑ | FPR@95 ↓ |
|:-------|----------:|--------:|---------:|
| k-NN (k=21, cosine) | 92.58% | 0.8211 | 0.5463 |
| Logistic Regression | 98.16% | 0.9702 | 0.1373 |
| **Proxy-Anchor (Ours)** | **98.23%** | **0.9798** | **0.0959** |

### Experiment 2 — Merged architectures (110 ID classes)

| Method | Accuracy ↑ | AUROC ↑ | FPR@95 ↓ |
|:-------|----------:|--------:|---------:|
| k-NN (k=21, cosine) | 95.15% | 0.7848 | 0.6690 |
| Logistic Regression | 99.59% | 0.9713 | 0.1651 |
| **Proxy-Anchor (Ours)** | **99.76%** | **0.9935** | **0.0204** |

## License

This project is licensed under the [MIT License](LICENSE).

## Contact
 
**Cristian-Teodor Neamtu** — cristian.neamtu [at] upb [dot] ro  
For questions about the code, please open an [issue](../../issues). 
For questions about the paper, feel free to reach out by email.

## Acknowledgements

This work was in part supported by a grant of the Ministry of Research, Innovation and Digitization, CCCDI - UEFISCDI, project number PN-IV-P7-7.1-PTE-2024-0600, within PNCDI IV. This work was also partially supported by the European Union – NextGenerationEU, through the National Recovery and Resilience Plan (PNRR), Component 9, Investment 4, under project SENSE, THINK @ POLITEHNICA BUCUREȘTI (SENTHIPoli) No. RUE 6.PI/I4/C9. The content of this material does not necessarily represent the official position of the European Union or the Government of Romania.
