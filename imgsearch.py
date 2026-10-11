#!/usr/bin/env python3
"""Search a folder of images by text or by example image using EmbeddingGemma 2.

Setup:
    pip install -U sentence-transformers transformers pillow numpy torch

Usage:
    python imgsearch.py index  /path/to/photos
    python imgsearch.py search /path/to/photos "dog playing in snow"
    python imgsearch.py search /path/to/photos --image query.jpg -k 5

The index is stored in <folder>/.imgsearch/ and is updated incrementally:
only new or modified images are embedded on later runs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "model/google/embeddinggemma-2"
DIM = 768  # may be 512 / 256 (128 hurts multimodal quality); must match at index and search time
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def load_model():
    # bf16 where supported; NEVER float16 (the model produces NaNs in fp16).
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    return SentenceTransformer(
        MODEL_ID,
        device=device,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"audio_config": None},  # skip the audio encoder; text+image only (~440M)
    )


def embed_image(model, path):
    # Images take no task prefix. <|image|> marks where the image sits in the input.
    emb = model.encode(
        {"text": "<|image|>", "image": [str(path)]},
        truncate_dim=DIM,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_text_query(model, query):
    emb = model.encode(query, prompt_name="SearchQuery",
                       truncate_dim=DIM, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def index_dir(folder):
    return Path(folder) / ".imgsearch"


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.suffix.lower() in EXTS and ".imgsearch" not in p.parts)


def load_index(folder):
    d = index_dir(folder)
    meta_f, vec_f = d / "meta.json", d / "vectors.npy"
    if meta_f.exists() and vec_f.exists():
        meta = json.loads(meta_f.read_text())
        if meta.get("dim") == DIM:
            return meta["files"], np.load(vec_f)
    return {}, np.zeros((0, DIM), dtype=np.float32)


def cmd_index(folder):
    folder = Path(folder).resolve()
    old_files, old_vecs = load_index(folder)  # rel path -> {"mtime", "row"}
    model = load_model()

    files, vecs = {}, []
    todo = []
    for p in list_images(folder):
        rel, mtime = str(p.relative_to(folder)), p.stat().st_mtime
        prev = old_files.get(rel)
        if prev and prev["mtime"] == mtime:
            files[rel] = {"mtime": mtime, "row": len(vecs)}
            vecs.append(old_vecs[prev["row"]])
        else:
            todo.append((rel, mtime, p))

    for i, (rel, mtime, p) in enumerate(todo, 1):
        try:
            v = embed_image(model, p)
        except Exception as e:  # corrupt/unsupported image
            print(f"skip {rel}: {e}", file=sys.stderr)
            continue
        files[rel] = {"mtime": mtime, "row": len(vecs)}
        vecs.append(v)
        print(f"[{i}/{len(todo)}] {rel}")

    d = index_dir(folder)
    d.mkdir(exist_ok=True)
    np.save(d / "vectors.npy", np.stack(vecs) if vecs else np.zeros((0, DIM), np.float32))
    (d / "meta.json").write_text(json.dumps({"dim": DIM, "files": files}))
    print(f"Indexed {len(files)} images ({len(todo)} new/changed).")


def cmd_search(folder, query, image, k):
    folder = Path(folder).resolve()
    files, vecs = load_index(folder)
    if not len(vecs):
        sys.exit("No index found. Run: python imgsearch.py index <folder>")
    model = load_model()

    q = embed_image(model, image) if image else embed_text_query(model, query)
    scores = vecs @ q  # cosine similarity (everything is L2-normalized)
    rel_by_row = {m["row"]: rel for rel, m in files.items()}
    for row in np.argsort(-scores)[:k]:
        print(f"{scores[row]:.3f}  {folder / rel_by_row[int(row)]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("folder")
    b = sub.add_parser("search"); b.add_argument("folder")
    b.add_argument("query", nargs="?"); b.add_argument("--image")
    b.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "index":
        cmd_index(args.folder)
    else:
        if not (args.query or args.image):
            ap.error("give a text query or --image")
        cmd_search(args.folder, args.query, args.image, args.k)


if __name__ == "__main__":
    main()