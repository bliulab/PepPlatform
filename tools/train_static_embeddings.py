import os
import sys
import argparse
import json
import hashlib
import tempfile
from contextlib import contextmanager
from collections import Counter
import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pepplatform.glove import glove_cache_signature, train_glove  # noqa: E402
from pepplatform.kmer import validate_kmer_k  # noqa: E402
from pepplatform.dataset import AA_VOCAB  # noqa: E402
from pepplatform.static_embeddings import (  # noqa: E402
    FASTTEXT_DEFAULT_BUCKET,
    FASTTEXT_MAX_N,
    FASTTEXT_MIN_N,
    STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
    ensure_fasttext_memory_budget,
    fasttext_cache_signature,
    stable_token_hash,
    validate_positive_int,
    validate_seed,
    validate_static_embedding_memory_limit_gib,
    word2vec_cache_signature,
)
from pepplatform.dr import (  # noqa: E402
    DR_DEFAULT_MAX_DISTANCE,
    DR_DEFAULT_MAX_RAW_LENGTH,
    dr_contract_metadata,
    iter_dr_tokens,
    normalize_dr_sequence,
    parse_dr_max_distance,
    parse_dr_max_raw_length,
    validate_dr_max_distance,
    validate_dr_max_raw_length,
)


def resolve_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def read_fasta_sequences(path):
    seqs = []
    cur = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line)
        if cur:
            seqs.append("".join(cur).upper())
    return seqs


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_kmer_sentences(sequences, k):
    k = validate_kmer_k(k)
    alphabet = {token for token in AA_VOCAB if len(token) == 1}
    sents = []
    for sequence_index, raw_sequence in enumerate(
        tqdm(sequences, desc=f"Building kmer-{k} sentences", leave=False),
        start=1,
    ):
        seq = "".join(str(raw_sequence or "").split()).upper()
        invalid = sorted(set(seq).difference(alphabet))
        if invalid:
            raise ValueError(
                f"K-mer static-embedding training sequence #{sequence_index} "
                f"contains unsupported character(s): {invalid}"
            )
        if len(seq) < k:
            continue
        toks = [seq[i : i + k] for i in range(len(seq) - k + 1)]
        if toks:
            sents.append(toks)
    return sents


def build_dr_sentences(
    sequences,
    max_distance,
    max_raw_length=DR_DEFAULT_MAX_RAW_LENGTH,
):
    sents = []
    max_distance = validate_dr_max_distance(max_distance)
    max_raw_length = validate_dr_max_raw_length(max_raw_length)
    for seq in tqdm(sequences, desc=f"Building dr-{max_distance} sentences", leave=False):
        normalized = normalize_dr_sequence(
            seq,
            max_raw_length=max_raw_length,
            context="DR static-embedding training sequence",
        )
        toks = list(iter_dr_tokens(normalized, max_distance))
        if toks:
            sents.append(toks)
    return sents


def _gensim_epoch_progress(epochs, desc):
    from gensim.models.callbacks import CallbackAny2Vec

    class _Callback(CallbackAny2Vec):
        def __init__(self):
            self.pbar = tqdm(total=epochs, desc=desc, leave=False)

        def on_epoch_end(self, model):
            self.pbar.update(1)

        def on_train_end(self, model):
            self.pbar.close()

    return _Callback()


def _validate_predictive_corpus(sentences, min_count, *, method):
    threshold = validate_positive_int(min_count, name=f"{method} min_count")
    if isinstance(sentences, (str, bytes)):
        raise ValueError(f"{method} corpus must be a sequence of token sequences.")
    normalized_sentences = []
    for sentence_index, sentence in enumerate(sentences):
        if isinstance(sentence, (str, bytes)):
            raise ValueError(
                f"{method} sentence {sentence_index} must be a token sequence, not text."
            )
        normalized = []
        for token_index, token in enumerate(sentence):
            if (
                not isinstance(token, str)
                or not token
                or any(character.isspace() for character in token)
            ):
                raise ValueError(
                    f"{method} token at sentence {sentence_index}, index "
                    f"{token_index} must be a non-empty string without whitespace."
                )
            normalized.append(token)
        normalized_sentences.append(tuple(normalized))
    frequencies = Counter(token for sentence in normalized_sentences for token in sentence)
    retained = {token for token, count in frequencies.items() if count >= threshold}
    if not retained:
        raise ValueError(f"No tokens left after min_count filtering for {method}.")
    context_pairs = 0
    for sentence in normalized_sentences:
        retained_count = sum(token in retained for token in sentence)
        if retained_count > 1:
            context_pairs += retained_count - 1
    if context_pairs == 0:
        raise ValueError(
            f"{method} corpus contains no trainable context pairs after min_count filtering."
        )
    return tuple(normalized_sentences), len(retained)


def _validate_gensim_vectors(model, *, method):
    vectors = model.wv.vectors
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise RuntimeError(f"{method} training produced an empty vector matrix.")
    if not bool(np.isfinite(vectors).all()):
        raise RuntimeError(f"{method} training produced non-finite vectors.")


def train_word2vec(sentences, vector_size, window, min_count, epochs, workers, seed=42):
    from gensim.models import Word2Vec

    sentences, _ = _validate_predictive_corpus(
        sentences,
        min_count,
        method="Word2Vec",
    )
    model = Word2Vec(
        sentences=sentences,
        vector_size=validate_positive_int(vector_size, name="Word2Vec vector_size"),
        window=validate_positive_int(window, name="Word2Vec window"),
        min_count=validate_positive_int(min_count, name="Word2Vec min_count"),
        workers=validate_positive_int(workers, name="Word2Vec workers"),
        sg=1,
        hs=0,
        negative=5,
        ns_exponent=0.75,
        sample=0.001,
        alpha=0.025,
        min_alpha=0.0001,
        shrink_windows=True,
        epochs=validate_positive_int(epochs, name="Word2Vec epochs"),
        callbacks=[_gensim_epoch_progress(int(epochs), "Training Word2Vec")],
        seed=validate_seed(seed),
        hashfxn=stable_token_hash,
    )
    _validate_gensim_vectors(model, method="Word2Vec")
    return model


def train_fasttext(
    sentences,
    vector_size,
    window,
    min_count,
    epochs,
    workers,
    seed=42,
    bucket=FASTTEXT_DEFAULT_BUCKET,
    memory_limit_gib=STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
):
    from gensim.models import FastText

    sentences, vocab_size = _validate_predictive_corpus(
        sentences,
        min_count,
        method="fastText",
    )
    ensure_fasttext_memory_budget(
        bucket=bucket,
        vector_size=vector_size,
        vocab_size=vocab_size,
        limit_gib=memory_limit_gib,
    )
    model = FastText(
        sentences=sentences,
        vector_size=validate_positive_int(vector_size, name="fastText vector_size"),
        window=validate_positive_int(window, name="fastText window"),
        min_count=validate_positive_int(min_count, name="fastText min_count"),
        workers=validate_positive_int(workers, name="fastText workers"),
        sg=1,
        hs=0,
        negative=5,
        ns_exponent=0.5,
        sample=0.0001,
        alpha=0.05,
        min_alpha=0.0,
        shrink_windows=True,
        epochs=validate_positive_int(epochs, name="fastText epochs"),
        min_n=FASTTEXT_MIN_N,
        max_n=FASTTEXT_MAX_N,
        bucket=validate_positive_int(bucket, name="fasttext_bucket"),
        callbacks=[_gensim_epoch_progress(int(epochs), "Training FastText")],
        seed=validate_seed(seed),
        hashfxn=stable_token_hash,
    )
    _validate_gensim_vectors(model, method="fastText")
    return model


def _atomic_output_path(out_path):
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(out_path)}.",
        suffix=".tmp",
        dir=out_dir,
    )
    os.close(descriptor)
    os.unlink(temporary)
    return temporary


@contextmanager
def cache_write_lock(out_dir):
    """Serialize artifact and metadata replacement for one cache directory."""
    os.makedirs(out_dir, exist_ok=True)
    lock_path = os.path.join(out_dir, ".write.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError(
                "static embedding cache locking requires fcntl on this platform"
            ) from exc
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def save_word2vec_vectors(model, out_path):
    temporary = _atomic_output_path(out_path)
    try:
        model.wv.save_word2vec_format(temporary, binary=False)
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_fasttext_model(model, out_path):
    from gensim.models.fasttext import save_facebook_model

    temporary = _atomic_output_path(out_path)
    try:
        save_facebook_model(model, temporary)
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_glove_vectors(vectors, out_path):
    temporary = _atomic_output_path(out_path)
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            for tok, vec in vectors.items():
                if any(character.isspace() for character in tok):
                    raise ValueError(f"GloVe token cannot contain whitespace: {tok!r}")
                if not bool(np.isfinite(vec).all()):
                    raise ValueError(f"GloVe vector for {tok!r} contains non-finite values")
                f.write(tok + " " + " ".join(f"{x:.8f}" for x in vec) + "\n")
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_embedding_dir(
    save_root,
    method,
    tokenizer_mode,
    k,
    dr_max_distance,
    cache_suffix="",
):
    if tokenizer_mode == "dr":
        tag = f"dr-v2-{validate_dr_max_distance(dr_max_distance)}"
    else:
        tag = f"kmer-{int(k)}"
    suffix = f"-{cache_suffix}" if cache_suffix else ""
    return os.path.join(save_root, f"{method}-{tag}{suffix}")


def save_cache_meta(out_dir, method, args, artifact_path, extra=None):
    meta = {
        "method": method,
        "tokenizer_mode": args.tokenizer_mode,
        "k": int(args.k),
        "dr_max_distance": int(args.dr_max_distance),
        "vector_size": int(args.vector_size),
        "window": int(args.window),
        "min_count": int(args.min_count),
        "epochs": int(args.epochs),
        "workers": 1 if method == "glove" else int(args.workers),
        "seed": int(args.seed),
        "feature_fit_scope": "train_only",
        "train_sha256": str(args.train_sha256),
        "artifact_file": os.path.basename(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_size_bytes": int(os.path.getsize(artifact_path)),
        "algorithm_verified": True,
        "verification": "internal_contract",
    }
    if args.tokenizer_mode == "dr":
        meta["dr_max_raw_length"] = validate_dr_max_raw_length(args.dr_max_raw_length)
        meta["dr_contract"] = dr_contract_metadata(
            max_distance=args.dr_max_distance,
            max_raw_length=args.dr_max_raw_length,
        )
    if args.cache_suffix:
        meta["cache_suffix"] = str(args.cache_suffix)
    if extra:
        meta.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".meta.",
        suffix=".json.tmp",
        dir=out_dir,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, meta_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(
        description="Train static embeddings (word2vec / GloVe / fasttext)"
    )
    parser.add_argument("--data_dir", type=str, default="datasets/task_peptide_function/data")
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="window length; tokenizer_mode=kmer accepts integers 1-3",
    )
    parser.add_argument("--tokenizer_mode", choices=["kmer", "dr"], default="kmer")
    parser.add_argument(
        "--cache_suffix",
        choices=["peptide", "partner", "protein"],
        default="",
    )
    parser.add_argument(
        "--dr_max_distance",
        type=parse_dr_max_distance,
        default=DR_DEFAULT_MAX_DISTANCE,
    )
    parser.add_argument(
        "--dr_max_raw_length",
        type=parse_dr_max_raw_length,
        default=DR_DEFAULT_MAX_RAW_LENGTH,
    )
    parser.add_argument("--vector_size", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fasttext_bucket", type=int, default=FASTTEXT_DEFAULT_BUCKET)
    parser.add_argument(
        "--static_embedding_memory_limit_gib",
        type=float,
        default=STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
    )
    parser.add_argument("--glove_x_max", type=float, default=100.0)
    parser.add_argument("--glove_alpha", type=float, default=0.75)
    parser.add_argument("--glove_learning_rate", type=float, default=0.05)
    parser.add_argument("--glove_batch_size", type=int, default=4096)
    parser.add_argument("--train_word2vec", action="store_true")
    parser.add_argument("--train_glove", action="store_true")
    parser.add_argument("--train_fasttext", action="store_true")
    parser.add_argument("--save_root", type=str, default="datasets")
    args = parser.parse_args()
    for name in ["vector_size", "window", "min_count", "epochs", "workers"]:
        try:
            setattr(
                args,
                name,
                validate_positive_int(getattr(args, name), name=name),
            )
        except ValueError as exc:
            parser.error(str(exc))
    try:
        args.seed = validate_seed(args.seed)
        args.fasttext_bucket = validate_positive_int(
            args.fasttext_bucket,
            name="fasttext_bucket",
        )
        args.static_embedding_memory_limit_gib = validate_static_embedding_memory_limit_gib(
            args.static_embedding_memory_limit_gib
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not (args.train_word2vec or args.train_glove or args.train_fasttext):
        args.train_word2vec = True
        args.train_glove = True
        args.train_fasttext = True

    data_dir = resolve_path(project_root, args.data_dir)
    save_root = resolve_path(project_root, args.save_root)

    fasta_paths = [os.path.join(data_dir, "train", "seqs.fasta")]
    missing = [p for p in fasta_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing fasta files: {missing}")

    sequences = []
    for p in fasta_paths:
        sequences.extend(read_fasta_sequences(p))
    args.train_sha256 = file_sha256(fasta_paths[0])
    if args.tokenizer_mode == "dr":
        sentences = build_dr_sentences(
            sequences,
            args.dr_max_distance,
            args.dr_max_raw_length,
        )
    else:
        sentences = build_kmer_sentences(sequences, args.k)
    if not sentences:
        raise SystemExit("No token sentences generated. Check tokenizer settings/data.")

    if args.train_word2vec or args.train_fasttext:
        try:
            import gensim  # noqa: F401
        except Exception:
            raise SystemExit("gensim is required for word2vec/fasttext. Run: pip install gensim")

    if args.train_word2vec:
        w2v_dir = resolve_embedding_dir(
            save_root,
            "word2vec",
            args.tokenizer_mode,
            args.k,
            args.dr_max_distance,
            args.cache_suffix,
        )
        w2v_path = os.path.join(w2v_dir, "word2vec.vec")
        model = train_word2vec(
            sentences=sentences,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs,
            workers=args.workers,
            seed=args.seed,
        )
        with cache_write_lock(w2v_dir):
            save_word2vec_vectors(model, w2v_path)
            save_cache_meta(
                w2v_dir,
                "word2vec",
                args,
                w2v_path,
                extra=word2vec_cache_signature(workers=args.workers),
            )
        print(f"Saved Word2Vec vectors to: {w2v_path}")

    if args.train_glove:
        glove_dir = resolve_embedding_dir(
            save_root, "glove", args.tokenizer_mode, args.k, args.dr_max_distance, args.cache_suffix
        )
        glove_path = os.path.join(glove_dir, "glove.vec")
        progress = tqdm(total=int(args.epochs), desc="Training GloVe", leave=False)

        def update_glove_progress(_epoch, loss):
            progress.set_postfix(loss=f"{loss:.6f}")
            progress.update(1)

        try:
            result = train_glove(
                sentences=sentences,
                vector_size=args.vector_size,
                window=args.window,
                min_count=args.min_count,
                epochs=args.epochs,
                seed=args.seed,
                x_max=args.glove_x_max,
                alpha=args.glove_alpha,
                learning_rate=args.glove_learning_rate,
                batch_size=args.glove_batch_size,
                epoch_callback=update_glove_progress,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        finally:
            progress.close()
        with cache_write_lock(glove_dir):
            save_glove_vectors(result.vectors, glove_path)
            save_cache_meta(
                glove_dir,
                "glove",
                args,
                glove_path,
                extra={
                    **glove_cache_signature(
                        x_max=args.glove_x_max,
                        alpha=args.glove_alpha,
                        learning_rate=args.glove_learning_rate,
                        batch_size=args.glove_batch_size,
                    ),
                    "vocab_size": int(result.vocab_size),
                    "nonzero_cooccurrences": int(result.nonzero_cooccurrences),
                    "initial_loss": float(result.initial_loss),
                    "final_loss": float(result.final_loss),
                },
            )
        print(f"Saved GloVe vectors to: {glove_path}")

    if args.train_fasttext:
        ft_dir = resolve_embedding_dir(
            save_root,
            "fasttext",
            args.tokenizer_mode,
            args.k,
            args.dr_max_distance,
            args.cache_suffix,
        )
        ft_path = os.path.join(ft_dir, "fasttext.bin")
        model = train_fasttext(
            sentences=sentences,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs,
            workers=args.workers,
            seed=args.seed,
            bucket=args.fasttext_bucket,
            memory_limit_gib=args.static_embedding_memory_limit_gib,
        )
        with cache_write_lock(ft_dir):
            save_fasttext_model(model, ft_path)
            save_cache_meta(
                ft_dir,
                "fasttext",
                args,
                ft_path,
                extra={
                    **fasttext_cache_signature(
                        workers=args.workers,
                        bucket=args.fasttext_bucket,
                    ),
                    "estimated_training_bytes": ensure_fasttext_memory_budget(
                        bucket=args.fasttext_bucket,
                        vector_size=args.vector_size,
                        vocab_size=len(model.wv.key_to_index),
                        limit_gib=args.static_embedding_memory_limit_gib,
                    ),
                    "static_embedding_memory_limit_gib": float(
                        args.static_embedding_memory_limit_gib
                    ),
                },
            )
        print(f"Saved FastText vectors to: {ft_path}")


if __name__ == "__main__":
    main()
