import torch
import inspect
from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    COVARIANCE_ENCODERS,
    validate_covariance_max_raw_length,
)
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_ENCODERS,
    validate_pdt_lag,
    validate_pdt_max_raw_length,
)
from pepplatform.kmer import validate_kmer_k
from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    FREQUENCY_ENCODER_CONTRACT_VERSION,
    build_frequency_encoder_contract,
    fit_smoothed_idf,
    normalize_frequency_sequence,
    validate_frequency_contract_version,
    validate_frequency_max_raw_length,
)
from pepplatform.dr import (
    DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    DR_DEFAULT_MAX_RAW_LENGTH,
    validate_dr_max_raw_length,
)
from pepplatform.textrank import build_textrank_contract
from pepplatform.tokenizers import (
    KmerTokenizer,
    DRTokenizer,
    TopNGramTokenizer,
    DTTokenizer,
    BPETokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
)
from pepplatform.encoders import (
    IndexEncoder,
    OneHotEncoder,
    DRCountEncoder,
    DRTfidfEncoder,
    DRTextRankEncoder,
    TopNGramCountEncoder,
    TopNGramTfidfEncoder,
    TopNGramTextRankEncoder,
    DTCountEncoder,
    DTTfidfEncoder,
    DTTextRankEncoder,
    BoWEncoder,
    TfidfEncoder,
    TextRankEncoder,
    AutoCovarianceEncoder,
    CrossCovarianceEncoder,
    AutoCrossCovarianceEncoder,
    PDTEncoder,
    Blosum62Encoder,
    PSSMEncoder,
    PSSMBlosum62HybridEncoder,
    PSFMEncoder,
    PSFMBlosum62HybridEncoder,
    ACPSSMEncoder,
    CCPSSMEncoder,
    ACCPSSMEncoder,
    ACPSSMBlosum62HybridEncoder,
    CCPSSMBlosum62HybridEncoder,
    ACCPSSMBlosum62HybridEncoder,
    Word2VecEncoder,
    GloveEncoder,
    FastTextEncoder,
)
from pepplatform.workflow import RESIDUE_PROFILE_ENCODERS, ensure_encoder_available
from pepplatform.wordpiece import WORDPIECE_DEFAULT_MAX_RAW_LENGTH
from pepplatform.unigram import (
    UNIGRAM_DEFAULT_ALPHA,
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    UNIGRAM_DEFAULT_NBEST_SIZE,
)


def _resolve_tfidf_idf(
    *,
    persisted_idf,
    documents,
    feature_dim,
    padding_id,
    context,
):
    document_list = [] if documents is None else list(documents)
    if persisted_idf is not None and document_list:
        raise ValueError(f"{context} accepts either training documents or persisted_idf, not both")
    if persisted_idf is not None:
        return persisted_idf, None
    idf, document_count = fit_smoothed_idf(
        document_list,
        feature_dim=feature_dim,
        padding_id=padding_id,
        context=context,
    )
    return idf, document_count


class FeaturizerFactory:
    def __init__(
        self,
        bpe_path,
        unigram_path=None,
        wordpiece_path=None,
        word2vec_path=None,
        glove_path=None,
        fasttext_path=None,
        pssm_index_path="",
        psfm_index_path="",
        ac_lag=10,
        cc_lag=10,
        acc_lag=10,
        pdt_lag=PDT_DEFAULT_LAG,
        ac_pssm_lag=10,
        cc_pssm_lag=10,
        acc_pssm_lag=10,
        wordpiece_max_raw_length=WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
        unigram_max_raw_length=UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
        dr_max_raw_length=DR_DEFAULT_MAX_RAW_LENGTH,
        frequency_profile_index_path="",
        covariance_max_raw_length=COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
        pdt_max_raw_length=PDT_DEFAULT_MAX_RAW_LENGTH,
        ac_pssm_blosum62_hybrid_index_path="",
        cc_pssm_blosum62_hybrid_index_path="",
        acc_pssm_blosum62_hybrid_index_path="",
        ac_pssm_blosum62_hybrid_lag=10,
        cc_pssm_blosum62_hybrid_lag=10,
        acc_pssm_blosum62_hybrid_lag=10,
    ):
        self.bpe_path = bpe_path
        self.unigram_path = unigram_path
        self.wordpiece_path = wordpiece_path
        self.word2vec_path = word2vec_path
        self.glove_path = glove_path
        self.fasttext_path = fasttext_path
        self.pssm_index_path = pssm_index_path
        self.psfm_index_path = psfm_index_path
        self.frequency_profile_index_path = frequency_profile_index_path
        self.ac_lag = ac_lag
        self.cc_lag = cc_lag
        self.acc_lag = acc_lag
        self.pdt_lag = validate_pdt_lag(pdt_lag)
        self.ac_pssm_lag = ac_pssm_lag
        self.cc_pssm_lag = cc_pssm_lag
        self.acc_pssm_lag = acc_pssm_lag
        self.ac_pssm_blosum62_hybrid_index_path = ac_pssm_blosum62_hybrid_index_path
        self.cc_pssm_blosum62_hybrid_index_path = cc_pssm_blosum62_hybrid_index_path
        self.acc_pssm_blosum62_hybrid_index_path = acc_pssm_blosum62_hybrid_index_path
        self.ac_pssm_blosum62_hybrid_lag = ac_pssm_blosum62_hybrid_lag
        self.cc_pssm_blosum62_hybrid_lag = cc_pssm_blosum62_hybrid_lag
        self.acc_pssm_blosum62_hybrid_lag = acc_pssm_blosum62_hybrid_lag
        self.wordpiece_max_raw_length = int(wordpiece_max_raw_length)
        self.unigram_max_raw_length = int(unigram_max_raw_length)
        self.dr_max_raw_length = validate_dr_max_raw_length(dr_max_raw_length)
        self.covariance_max_raw_length = validate_covariance_max_raw_length(
            covariance_max_raw_length
        )
        self.pdt_max_raw_length = validate_pdt_max_raw_length(pdt_max_raw_length)

    def create(
        self,
        tokenizer,
        encoder,
        max_length,
        k,
        sequences=None,
        dr_max_distance=5,
        dr_unknown_residue_policy=DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
        top_ngram_n=2,
        dt_max_distance=5,
        profile_samples=None,
        persisted_idf=None,
        frequency_contract_version=FREQUENCY_ENCODER_CONTRACT_VERSION,
        frequency_max_raw_length=FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    ):
        ensure_encoder_available(encoder)
        if encoder in RESIDUE_PROFILE_ENCODERS and tokenizer != "kmer":
            raise ValueError(f"{encoder} requires tokenizer=kmer and k=1")
        frequency_contract_version = validate_frequency_contract_version(frequency_contract_version)
        frequency_max_raw_length = validate_frequency_max_raw_length(frequency_max_raw_length)
        frequency_v2 = frequency_contract_version == FREQUENCY_ENCODER_CONTRACT_VERSION

        def finish_frequency(tok, enc, *, training_document_count=None):
            result = ComposedFeaturizer(tok, enc)
            result.frequency_encoder_contract = build_frequency_encoder_contract(
                tokenizer=tok,
                encoder=encoder,
                contract_version=frequency_contract_version,
                max_raw_length=frequency_max_raw_length,
                training_document_count=training_document_count,
            )
            return result

        def finish_textrank(tok, enc, *, stream_policy):
            result = ComposedFeaturizer(tok, enc)
            result.textrank_contract = build_textrank_contract(
                feature_dim=enc.feature_dim,
                stream_policy=stream_policy,
            )
            return result

        def finish_covariance(tok, enc):
            result = ComposedFeaturizer(tok, enc)
            result.covariance_encoder_contract = dict(enc.covariance_encoder_contract)
            return result

        def finish_pdt(tok, enc):
            result = ComposedFeaturizer(tok, enc)
            result.pdt_encoder_contract = dict(enc.pdt_encoder_contract)
            return result

        def finish_pssm_covariance(tok, enc):
            result = ComposedFeaturizer(tok, enc)
            result.pssm_covariance_encoder_contract = dict(enc.pssm_covariance_encoder_contract)
            return result

        if tokenizer == "kmer":
            normalized_k = validate_kmer_k(k)
            if encoder in RESIDUE_PROFILE_ENCODERS and normalized_k != 1:
                raise ValueError(
                    f"{encoder} is defined per raw residue and requires k=1; "
                    "PSSM/PSFM profile k-mer aggregation is not supported"
                )
            if encoder in COVARIANCE_ENCODERS and normalized_k != 1:
                raise ValueError(f"{encoder} is defined per raw residue and requires k=1")
            if encoder in PDT_ENCODERS and normalized_k != 1:
                raise ValueError("pdt is defined per raw residue and requires k=1")
            tok = KmerTokenizer(
                max_length=max_length,
                k=k,
                frequency_max_raw_length=frequency_max_raw_length,
            )
            if encoder == "index":
                enc = IndexEncoder(
                    max_length=max_length, vocab_size=tok.vocab_size, to_indices=tok.to_indices
                )
            elif encoder == "onehot":
                enc = OneHotEncoder(
                    max_length=max_length, vocab_size=tok.vocab_size, to_indices=tok.to_indices
                )
            elif encoder == "bow":
                enc = BoWEncoder(
                    max_length=max_length,
                    vocab_size=tok.vocab_size,
                    to_indices=tok.to_indices,
                    normalize=not frequency_v2,
                    sequence_to_ids=(tok.iter_frequency_feature_ids if frequency_v2 else None),
                )
                return finish_frequency(tok, enc)
            elif encoder in ["tfidf", "TF-IDF"]:
                sequence_list = [] if sequences is None else list(sequences)
                if frequency_v2:
                    documents = (tok.iter_frequency_feature_ids(seq) for seq in sequence_list)
                else:
                    documents = (tok.to_indices(tok.tokenize(seq)) for seq in sequence_list)
                idf, document_count = _resolve_tfidf_idf(
                    persisted_idf=persisted_idf,
                    documents=documents,
                    feature_dim=tok.vocab_size,
                    padding_id=0,
                    context="K-mer TF-IDF",
                )
                enc = TfidfEncoder(
                    max_length=max_length,
                    vocab_size=tok.vocab_size,
                    to_indices=tok.to_indices,
                    idf=idf,
                    normalize=True,
                    sequence_to_ids=(tok.iter_frequency_feature_ids if frequency_v2 else None),
                )
                return finish_frequency(
                    tok,
                    enc,
                    training_document_count=document_count,
                )
            elif encoder == "textrank":
                enc = TextRankEncoder(
                    max_length=max_length, vocab_size=tok.vocab_size, to_indices=tok.to_indices
                )
            elif encoder == "ac":
                enc = AutoCovarianceEncoder(
                    lag=self.ac_lag,
                    max_raw_length=self.covariance_max_raw_length,
                )
                return finish_covariance(tok, enc)
            elif encoder == "cc":
                enc = CrossCovarianceEncoder(
                    lag=self.cc_lag,
                    max_raw_length=self.covariance_max_raw_length,
                )
                return finish_covariance(tok, enc)
            elif encoder == "acc":
                enc = AutoCrossCovarianceEncoder(
                    lag=self.acc_lag,
                    max_raw_length=self.covariance_max_raw_length,
                )
                return finish_covariance(tok, enc)
            elif encoder == "pdt":
                enc = PDTEncoder(
                    lag=self.pdt_lag,
                    max_raw_length=self.pdt_max_raw_length,
                )
                return finish_pdt(tok, enc)
            elif encoder == "blosum62":
                enc = Blosum62Encoder(max_length=max_length)
            elif encoder == "pssm":
                if not self.pssm_index_path:
                    raise ValueError("pssm requires --pssm_index_path")
                enc = PSSMEncoder(
                    max_length=max_length, index_path=self.pssm_index_path, strict=True
                )
            elif encoder == "pssm_blosum62_hybrid":
                if not self.pssm_index_path:
                    raise ValueError("pssm_blosum62_hybrid requires --pssm_index_path")
                enc = PSSMBlosum62HybridEncoder(
                    max_length=max_length,
                    index_path=self.pssm_index_path,
                    strict=True,
                )
            elif encoder == "ac_pssm":
                if not self.pssm_index_path:
                    raise ValueError("ac_pssm requires --pssm_index_path")
                enc = ACPSSMEncoder(
                    max_length=max_length,
                    index_path=self.pssm_index_path,
                    lag=self.ac_pssm_lag,
                    strict=True,
                )
                return finish_pssm_covariance(tok, enc)
            elif encoder == "cc_pssm":
                if not self.pssm_index_path:
                    raise ValueError("cc_pssm requires --pssm_index_path")
                enc = CCPSSMEncoder(
                    max_length=max_length,
                    index_path=self.pssm_index_path,
                    lag=self.cc_pssm_lag,
                    strict=True,
                )
                return finish_pssm_covariance(tok, enc)
            elif encoder == "acc_pssm":
                if not self.pssm_index_path:
                    raise ValueError("acc_pssm requires --pssm_index_path")
                enc = ACCPSSMEncoder(
                    max_length=max_length,
                    index_path=self.pssm_index_path,
                    lag=self.acc_pssm_lag,
                    strict=True,
                )
                return finish_pssm_covariance(tok, enc)
            elif encoder == "ac_pssm_blosum62_hybrid":
                if not self.ac_pssm_blosum62_hybrid_index_path:
                    raise ValueError(
                        "ac_pssm_blosum62_hybrid requires --ac_pssm_blosum62_hybrid_index_path"
                    )
                enc = ACPSSMBlosum62HybridEncoder(
                    max_length=max_length,
                    index_path=self.ac_pssm_blosum62_hybrid_index_path,
                    lag=self.ac_pssm_blosum62_hybrid_lag,
                    strict=True,
                )
                return finish_pssm_covariance(tok, enc)
            elif encoder == "cc_pssm_blosum62_hybrid":
                if not self.cc_pssm_blosum62_hybrid_index_path:
                    raise ValueError(
                        "cc_pssm_blosum62_hybrid requires --cc_pssm_blosum62_hybrid_index_path"
                    )
                enc = CCPSSMBlosum62HybridEncoder(
                    max_length=max_length,
                    index_path=self.cc_pssm_blosum62_hybrid_index_path,
                    lag=self.cc_pssm_blosum62_hybrid_lag,
                    strict=True,
                )
                return finish_pssm_covariance(tok, enc)
            elif encoder == "acc_pssm_blosum62_hybrid":
                if not self.acc_pssm_blosum62_hybrid_index_path:
                    raise ValueError(
                        "acc_pssm_blosum62_hybrid requires --acc_pssm_blosum62_hybrid_index_path"
                    )
                enc = ACCPSSMBlosum62HybridEncoder(
                    max_length=max_length,
                    index_path=self.acc_pssm_blosum62_hybrid_index_path,
                    lag=self.acc_pssm_blosum62_hybrid_lag,
                    strict=True,
                )
                return finish_pssm_covariance(tok, enc)
            elif encoder == "psfm":
                if not self.psfm_index_path:
                    raise ValueError("psfm requires --psfm_index_path")
                enc = PSFMEncoder(
                    max_length=max_length, index_path=self.psfm_index_path, strict=True
                )
            elif encoder == "psfm_blosum62_hybrid":
                if not self.psfm_index_path:
                    raise ValueError("psfm_blosum62_hybrid requires --psfm_index_path")
                enc = PSFMBlosum62HybridEncoder(
                    max_length=max_length,
                    index_path=self.psfm_index_path,
                    strict=True,
                )
            elif encoder == "word2vec":
                if not self.word2vec_path:
                    raise ValueError("word2vec requires --word2vec_path")
                enc = Word2VecEncoder(max_length=max_length, embedding_path=self.word2vec_path)
            elif encoder == "glove":
                if not self.glove_path:
                    raise ValueError("glove requires --glove_path")
                enc = GloveEncoder(max_length=max_length, embedding_path=self.glove_path)
            elif encoder == "fasttext":
                if not self.fasttext_path:
                    raise ValueError("fasttext requires --fasttext_path")
                enc = FastTextEncoder(max_length=max_length, embedding_path=self.fasttext_path)
            else:
                raise ValueError("unsupported encoder")
            if encoder == "textrank":
                return finish_textrank(
                    tok,
                    enc,
                    stream_policy="tokenizer_emitted_truncated_kmer_stream",
                )
            return ComposedFeaturizer(tok, enc)
        if tokenizer == "dr":
            tok = DRTokenizer(
                max_length=max_length,
                max_distance=dr_max_distance,
                unknown_residue_policy=dr_unknown_residue_policy,
                max_raw_length=(
                    frequency_max_raw_length
                    if frequency_v2 and encoder in {"bow", "tfidf", "TF-IDF"}
                    else self.dr_max_raw_length
                ),
            )
            if encoder == "index":
                enc = IndexEncoder(
                    max_length=max_length, vocab_size=tok.vocab_size, to_indices=tok.to_indices
                )
            elif encoder == "onehot":
                enc = OneHotEncoder(
                    max_length=max_length, vocab_size=tok.vocab_size, to_indices=tok.to_indices
                )
            elif encoder == "bow":
                enc = DRCountEncoder(tok)
                return finish_frequency(tok, enc)
            elif encoder in ["tfidf", "TF-IDF"]:
                sequence_list = [] if sequences is None else list(sequences)
                idf, document_count = _resolve_tfidf_idf(
                    persisted_idf=persisted_idf,
                    documents=(
                        (tok.feature_to_index[token] for token in tok.iter_tokens(seq))
                        for seq in sequence_list
                    ),
                    feature_dim=tok.feature_dim,
                    padding_id=None,
                    context="DR TF-IDF",
                )
                enc = DRTfidfEncoder(tok, idf=idf)
                return finish_frequency(
                    tok,
                    enc,
                    training_document_count=document_count,
                )
            elif encoder == "textrank":
                enc = DRTextRankEncoder(tok)
            elif encoder == "word2vec":
                if not self.word2vec_path:
                    raise ValueError("word2vec requires --word2vec_path")
                enc = Word2VecEncoder(max_length=max_length, embedding_path=self.word2vec_path)
            elif encoder == "glove":
                if not self.glove_path:
                    raise ValueError("glove requires --glove_path")
                enc = GloveEncoder(max_length=max_length, embedding_path=self.glove_path)
            elif encoder == "fasttext":
                if not self.fasttext_path:
                    raise ValueError("fasttext requires --fasttext_path")
                enc = FastTextEncoder(max_length=max_length, embedding_path=self.fasttext_path)
            else:
                raise ValueError(
                    "dr tokenizer supports encoder in {index,onehot,bow,tfidf,textrank,word2vec,glove,fasttext}"
                )
            if encoder == "textrank":
                return finish_textrank(
                    tok,
                    enc,
                    stream_policy="complete_normalized_source_position_major_dr_stream",
                )
            return ComposedFeaturizer(tok, enc)
        if tokenizer == "top-n-gram":
            if not self.frequency_profile_index_path:
                legacy_hint = (
                    " A legacy --pssm_index_path is configured, but PSSM-score "
                    "indexes are intentionally incompatible."
                    if self.pssm_index_path
                    else ""
                )
                raise ValueError(
                    "top-n-gram v3 requires --frequency_profile_index_path." + legacy_hint
                )
            tok = TopNGramTokenizer(
                index_path=self.frequency_profile_index_path,
                max_length=max_length,
                top_n=top_ngram_n,
                strict=True,
                frequency_max_raw_length=(
                    frequency_max_raw_length
                    if frequency_v2 and encoder in {"bow", "tfidf", "TF-IDF"}
                    else None
                ),
            )
            if encoder == "bow":
                enc = TopNGramCountEncoder(
                    tok,
                    max_raw_length=frequency_max_raw_length,
                )
                return finish_frequency(tok, enc)
            elif encoder in ["tfidf", "TF-IDF"]:
                samples = [] if profile_samples is None else list(profile_samples)
                documents = []
                for sample in samples:
                    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                        raise ValueError(
                            "Top-n-gram profile_samples entries must be (sample_id, sequence) pairs"
                        )
                    sample_id, sequence = sample
                    sequence = normalize_frequency_sequence(
                        sequence,
                        max_raw_length=frequency_max_raw_length,
                    )
                    documents.append(tok.feature_ids(sequence, sample_id=sample_id))
                idf, document_count = _resolve_tfidf_idf(
                    persisted_idf=persisted_idf,
                    documents=documents,
                    feature_dim=tok.feature_dim,
                    padding_id=None,
                    context="Top-n-gram TF-IDF",
                )
                enc = TopNGramTfidfEncoder(
                    tok,
                    idf=idf,
                    max_raw_length=frequency_max_raw_length,
                )
                return finish_frequency(
                    tok,
                    enc,
                    training_document_count=document_count,
                )
            elif encoder == "textrank":
                enc = TopNGramTextRankEncoder(tok)
            else:
                raise ValueError("top-n-gram tokenizer supports encoder in {bow,tfidf,textrank}")
            if encoder == "textrank":
                return finish_textrank(
                    tok,
                    enc,
                    stream_policy="complete_profile_residue_top_n_gram_stream",
                )
            return ComposedFeaturizer(tok, enc)
        if tokenizer == "dt":
            if not self.frequency_profile_index_path:
                raise ValueError("DT v1 requires --frequency_profile_index_path")
            tok = DTTokenizer(
                index_path=self.frequency_profile_index_path,
                max_length=max_length,
                max_distance=dt_max_distance,
                strict=True,
                frequency_max_raw_length=(
                    frequency_max_raw_length
                    if frequency_v2 and encoder in {"bow", "tfidf", "TF-IDF"}
                    else None
                ),
            )
            if encoder == "bow":
                enc = DTCountEncoder(
                    tok,
                    max_raw_length=frequency_max_raw_length,
                )
                return finish_frequency(tok, enc)
            elif encoder in ["tfidf", "TF-IDF"]:
                samples = [] if profile_samples is None else list(profile_samples)
                documents = []
                for sample in samples:
                    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                        raise ValueError(
                            "DT profile_samples entries must be (sample_id, sequence) pairs"
                        )
                    sample_id, sequence = sample
                    sequence = normalize_frequency_sequence(
                        sequence,
                        max_raw_length=frequency_max_raw_length,
                    )
                    documents.append(tok.iter_feature_ids(sequence, sample_id=sample_id))
                idf, document_count = _resolve_tfidf_idf(
                    persisted_idf=persisted_idf,
                    documents=documents,
                    feature_dim=tok.feature_dim,
                    padding_id=None,
                    context="DT TF-IDF",
                )
                enc = DTTfidfEncoder(
                    tok,
                    idf=idf,
                    max_raw_length=frequency_max_raw_length,
                )
                return finish_frequency(
                    tok,
                    enc,
                    training_document_count=document_count,
                )
            elif encoder == "textrank":
                enc = DTTextRankEncoder(tok)
            else:
                raise ValueError("dt tokenizer supports encoder in {bow,tfidf,textrank}")
            if encoder == "textrank":
                return finish_textrank(
                    tok,
                    enc,
                    stream_policy="complete_source_position_major_dt_stream",
                )
            return ComposedFeaturizer(tok, enc)
        if tokenizer == "bpe":
            if encoder != "index":
                raise ValueError("bpe currently requires encoder=index")
            tok = BPETokenizer(tokenizer_path=self.bpe_path, max_length=max_length)
            enc = IndexEncoder(max_length=max_length, vocab_size=tok.vocab_size)
            return ComposedFeaturizer(tok, enc)
        if tokenizer == "wordpiece":
            if encoder != "index":
                raise ValueError("wordpiece currently requires encoder=index")
            if not self.wordpiece_path:
                raise ValueError("wordpiece requires wordpiece_path")
            tok = WordPieceTokenizer(
                tokenizer_path=self.wordpiece_path,
                max_length=max_length,
                max_raw_length=self.wordpiece_max_raw_length,
            )
            enc = IndexEncoder(max_length=max_length, vocab_size=tok.vocab_size)
            return ComposedFeaturizer(tok, enc)
        if tokenizer == "unigram":
            if encoder != "index":
                raise ValueError("unigram currently requires encoder=index")
            if not self.unigram_path:
                raise ValueError("unigram requires unigram_path")
            tok = UnigramTokenizer(
                tokenizer_path=self.unigram_path,
                max_length=max_length,
                max_raw_length=self.unigram_max_raw_length,
            )
            enc = IndexEncoder(max_length=max_length, vocab_size=tok.vocab_size)
            return ComposedFeaturizer(tok, enc)
        raise ValueError("unsupported tokenizer")


class ComposedFeaturizer:
    def __init__(self, tokenizer, encoder):
        self.max_length = tokenizer.max_length
        self.tokenizer = tokenizer
        self.encoder = encoder
        self._supports_sample_id = "sample_id" in inspect.signature(self.encoder.encode).parameters
        self._tokenizer_supports_sample_id = (
            "sample_id" in inspect.signature(self.tokenizer.tokenize).parameters
        )
        self._supports_raw_sequence = hasattr(self.encoder, "encode_sequence")
        self._raw_sequence_supports_sample_id = (
            self._supports_raw_sequence
            and "sample_id" in inspect.signature(self.encoder.encode_sequence).parameters
        )
        self.explicit_position_mask = False
        if hasattr(encoder, "vocab_size"):
            self.vocab_size = encoder.vocab_size
        if hasattr(encoder, "feature_dim"):
            self.feature_dim = encoder.feature_dim
        if hasattr(encoder, "feature_names"):
            self.feature_names = tuple(encoder.feature_names)

    def encode(self, sequence: str, sample_id=None) -> torch.Tensor:
        if bool(getattr(self.encoder, "requires_explicit_mask", False)):
            from pepplatform.models import ResidueModelInput

            features, mask = self.encoder.encode_sequence_with_mask(
                sequence,
                sample_id=sample_id,
            )
            return ResidueModelInput(features, mask)
        if self._supports_raw_sequence:
            if self.explicit_position_mask and hasattr(
                self.encoder,
                "encode_sequence_with_mask",
            ):
                from pepplatform.models import ResidueModelInput

                features, mask = self.encoder.encode_sequence_with_mask(
                    sequence,
                    sample_id=sample_id,
                )
                return ResidueModelInput(features, mask)
            if self._raw_sequence_supports_sample_id:
                return self.encoder.encode_sequence(
                    sequence,
                    sample_id=sample_id,
                )
            return self.encoder.encode_sequence(sequence)
        if hasattr(self.tokenizer, "tokenize_for_ordered_encoder"):
            tokens = self.tokenizer.tokenize_for_ordered_encoder(sequence)
        elif self._tokenizer_supports_sample_id:
            tokens = self.tokenizer.tokenize(sequence, sample_id=sample_id)
        else:
            tokens = self.tokenizer.tokenize(sequence)
        if self._supports_sample_id:
            encoded = self.encoder.encode(tokens, sample_id=sample_id)
        else:
            encoded = self.encoder.encode(tokens)
        if self.explicit_position_mask and isinstance(encoded, torch.Tensor) and encoded.ndim == 2:
            from pepplatform.models import ResidueModelInput

            valid_count = min(len(tokens), int(encoded.shape[0]))
            mask = torch.zeros(encoded.shape[0], dtype=torch.bool)
            mask[:valid_count] = True
            return ResidueModelInput(encoded, mask)
        return encoded

    def require_explicit_position_mask(self):
        self.explicit_position_mask = True
        return self

    @property
    def supports_unigram_sampling(self):
        return hasattr(self.tokenizer, "tokenize_sampled")

    def encode_sampled(
        self,
        sequence: str,
        *,
        seed: int,
        epoch: int,
        sample_id,
        role="sequence",
        nbest_size=UNIGRAM_DEFAULT_NBEST_SIZE,
        alpha=UNIGRAM_DEFAULT_ALPHA,
    ) -> torch.Tensor:
        if not self.supports_unigram_sampling:
            raise ValueError("featurizer tokenizer does not support Unigram sampling")
        tokens = self.tokenizer.tokenize_sampled(
            sequence,
            seed=seed,
            epoch=epoch,
            sample_id=sample_id,
            role=role,
            nbest_size=nbest_size,
            alpha=alpha,
        )
        if self._supports_sample_id:
            return self.encoder.encode(tokens, sample_id=sample_id)
        return self.encoder.encode(tokens)
