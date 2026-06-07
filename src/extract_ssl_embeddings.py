import argparse
import logging
import os
import time
import warnings
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

TARGET_SR = 16000
DEFAULT_LAYER = 4
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )


def load_model(model_name, device):

    print(f"Loading {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    model.eval()
    model.to(device)

    embedding_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    print(f"Model loaded on {device}")
    print(f"  Hidden size: {embedding_dim}, Transformer layers: {num_layers}")
    print(
        f"  hidden_states indices: 0 (CNN/feature extractor) + 1..{num_layers} (transformer layers)"
    )

    return feature_extractor, model, embedding_dim, num_layers


def discover_audio_files(input_dir, output_dir, no_resume=False):

    output_dir = Path(output_dir)

    to_process = []
    total_found = 0
    total_skipped = 0

    for lang_dir in sorted(input_dir.iterdir()):
        if not lang_dir.is_dir():
            continue

        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            language = lang_dir.name
            model_name = model_dir.name

            for audio_file in sorted(model_dir.iterdir()):
                if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue

                total_found += 1
                output_path = (
                    output_dir / language / model_name / f"{audio_file.stem}.pt"
                )

                if not no_resume and output_path.exists():
                    total_skipped += 1
                    continue

                to_process.append((audio_file, output_path))

    return to_process, total_found, total_skipped


def extract_single(audio_path, feature_extractor, model, device, layer, embedding_dim):
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR)

    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    layer_output = outputs.hidden_states[layer]
    embedding = layer_output.mean(dim=1).squeeze(0).cpu()

    return embedding


def process_files(to_process, feature_extractor, model, device, layer, embedding_dim):
    num_success = 0
    num_failed = 0

    for audio_path, output_path in tqdm(to_process, desc="Extracting embeddings"):
        try:
            embedding = extract_single(
                audio_path, feature_extractor, model, device, layer, embedding_dim
            )

            assert embedding.shape == (
                embedding_dim,
            ), f"Expected shape ({embedding_dim},), got {embedding.shape}"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding, output_path)
            num_success += 1

        except Exception as e:
            num_failed += 1
            logging.error(f"Failed: {audio_path} — {e}")

    return num_success, num_failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from any HuggingFace SSL speech model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "HuggingFace model name, e.g. 'microsoft/wavlm-large', "
            "'facebook/hubert-large-ll60k', 'facebook/w2v-bert-2.0'"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory of audio files: {input_dir}/{language}/{model_name}/{audio}.wav",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for output .pt files: {output_dir}/{language}/{model_name}/{audio}.pt",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=(
            f"Transformer layer index in hidden_states to extract (default: {DEFAULT_LAYER}). "
            "Index 0 = CNN/feature extractor output; 1..N = transformer layer outputs. "
            "Index 0 = CNN/feature extractor output; 1..N = transformer layer outputs. "
            "Layer 4 is used for all models to keep the comparison controlled."
        ),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to error log file (default: {output_dir}/extraction_errors.log)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Re-extract all files even if output already exists",
    )

    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(args.output_dir, "extraction_errors.log")
    setup_logging(log_file)

    feature_extractor, model, embedding_dim, num_layers = load_model(
        args.model_name, device
    )

    max_layer = num_layers
    if args.layer > max_layer:
        logging.warning(
            f"Requested layer {args.layer} exceeds model depth ({num_layers} transformer layers, "
            f"max hidden_states index = {max_layer}). Falling back to layer {max_layer}."
        )
        layer = max_layer
    else:
        layer = args.layer

    print(f"\nExtracting layer {layer} (embedding_dim={embedding_dim})")
    print(f"Scanning {args.input_dir}...")

    to_process, total_found, total_skipped = discover_audio_files(
        args.input_dir, args.output_dir, args.no_resume
    )

    print(f"  Total audio files found: {total_found}")
    print(f"  Already extracted (skipped): {total_skipped}")
    print(f"  To process: {len(to_process)}")

    if not to_process:
        print("\nNothing to process.")
        return

    start_time = time.time()
    num_success, num_failed = process_files(
        to_process, feature_extractor, model, device, layer, embedding_dim
    )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Success: {num_success}")
    print(f"  Failed:  {num_failed}")

    if num_failed > 0:
        print(f"  See {log_file} for error details")


if __name__ == "__main__":
    main()
