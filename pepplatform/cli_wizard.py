from pepplatform.task_compat import normalize_task_type, resolve_hlai_root

from pepplatform.pair_terminology import normalize_pair_strategy, public_pair_strategy

import json
import math
import os
from pepplatform.kmer import KMER_MAX_K
from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    FREQUENCY_MAX_RAW_LENGTH,
)
from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    COVARIANCE_ENCODERS,
    COVARIANCE_MAX_RAW_LENGTH,
)
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_MAX_RAW_LENGTH,
)
from pepplatform.pssm_covariance import (
    HYBRID_PSSM_COVARIANCE_ENCODERS,
    PSSM_COVARIANCE_ENCODERS,
    pssm_covariance_lag_key,
)
from pepplatform.dr import (
    DR_ABSOLUTE_MAX_RAW_LENGTH,
    DR_DEFAULT_MAX_RAW_LENGTH,
    DR_MAX_DISTANCE,
)
from pepplatform.dt import DT_MAX_DISTANCE
from pepplatform.top_ngram import TOP_N_GRAM_FALLBACK_POLICY
from pepplatform.lsa import LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.lda import LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.plsa import PLSA_DEFAULT_MEMORY_LIMIT_GIB
from pepplatform.esmc import official_esmc_model_options
from pepplatform.workflow import (
    CHAIN_GRAPH_ARCHITECTURES,
    CHAIN_GRAPH_COMPATIBLE_ENCODERS,
    CLASSICAL_CLASSIFIERS,
    CLASSIFIERS,
    END_TO_END,
    EXPORT_EMBEDDINGS,
    GRAPH_PREDICTORS,
    PAIR_INTERACTION_MODELS,
    PRETRAINED_MODEL_DEFAULTS,
    RESIDUE_ADAPTER_DIM_DEFAULT,
    RESIDUE_ADAPTER_DIM_MAX,
    RESIDUE_ADAPTER_DIM_MIN,
    RESIDUE_ADAPTER_DIM_MULTIPLE,
    SEQUENCE_ARCHITECTURES,
    SEQUENCE_MODELS,
    TOKEN_INTERACTION,
    VECTOR_INTERACTION,
    effective_classifier,
    encoder_family,
    encoder_options_for_tokenizer,
    normalize_tokenizer_name,
    normalize_pair_workflow_args,
    normalize_workflow_args,
    default_prediction_head_hidden_dims,
    parse_prediction_head_hidden_dims,
    prediction_head_applies,
    transformation_options_for_encoder,
)

DL_SEQUENCE_MODEL_NAMES = SEQUENCE_MODELS
PAIR_TOKEN_INTERACTION_MODEL_NAMES = [
    "cross_attn",
    "cross_attn+mlp",
    "bi_cross_attn",
    "bi_cross_attn+mlp",
]
PAIR_SEMANTIC_MODEL_NAMES = [
    model for model in DL_SEQUENCE_MODEL_NAMES if model not in GRAPH_PREDICTORS
] + ["mlp", "cross_attn", "cross_attn+mlp", "bi_cross_attn", "bi_cross_attn+mlp"]
TRAINABLE_DL_MODEL_NAMES = DL_SEQUENCE_MODEL_NAMES + ["mlp"] + PAIR_TOKEN_INTERACTION_MODEL_NAMES


class InteractiveConfigWizard:
    def _task_root(self, task_type):
        task_type = normalize_task_type(task_type)
        normalized = normalize_task_type(task_type or "pep_function")
        if normalized == "pep_function_multilabel":
            normalized = "pep_function"
        if normalized == "pep_hlai":
            return os.path.relpath(resolve_hlai_root(self._project_root()), self._project_root())
        task_root_map = {
            "pep_function": "datasets/task_peptide_function",
            "pep_protein_interaction": "datasets/task_peptide_protein_interaction",
            "pep_hlaii": "datasets/task_peptide_HLAII_interaction",
            "pep_hlai": "datasets/task_peptide_HLAI_interaction",
        }
        return task_root_map.get(normalized, f"datasets/task_{normalized}")

    def _project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _path_exists(self, path):
        if not path:
            return False
        if os.path.exists(path):
            return True
        if os.path.isabs(path):
            return False
        return os.path.exists(os.path.join(self._project_root(), path))

    def _default_data_dir(self, args, task_root):
        explicit_data_dir = getattr(args, "data_dir", "")
        if explicit_data_dir:
            return explicit_data_dir
        if normalize_task_type(getattr(args, "task_type", "")) == "pep_hlaii":
            quick_dir = f"{task_root}/data_quick_50k"
            if self._path_exists(quick_dir):
                return quick_dir
        return f"{task_root}/data"

    def _default_pretrained_feature_dir(self, args, task_root, feature_name):
        data_dir = self._default_data_dir(args, task_root)
        if (
            normalize_task_type(getattr(args, "task_type", "")) == "pep_hlaii"
            and os.path.basename(os.path.normpath(data_dir)) == "data_quick_50k"
        ):
            return f"{data_dir}/{feature_name}_features"
        return f"{task_root}/{feature_name}_features"

    def _fusion_options_for_model(self, model):
        if str(model).strip().lower() == "hgt":
            return []
        options = ["concat_all", "concat", "concat_absdiff", "concat_hadamard", "gated(static)"]
        if str(model).strip().lower() in ["mlp", "gcn", "gat"]:
            options.append("aligned_gated")
        return options

    def _fusion_default(self, args, options):
        fusion = str(getattr(args, "fusion_mode", "concat_all")).strip().lower()
        if fusion == "gated":
            fusion = "gated(static)"
        elif fusion in ["concat_semantic", "semantic", "concat+semantic"]:
            fusion = "concat_all"
        return fusion if fusion in options else "concat_all"

    def _apply_fusion_choice(self, args, choice):
        if choice == "gated(static)":
            args.fusion_mode = "gated"
        else:
            args.fusion_mode = choice

    def _set_classifier(self, args, classifier):
        args.sequence_model = ""
        args.training_mode = ""
        args.classifier = classifier
        args.model = classifier

    def _set_sequence_route(self, args, sequence_model, training_mode, classifier="mlp"):
        args.sequence_model = sequence_model
        args.training_mode = training_mode
        args.classifier = classifier
        args.model = sequence_model

    def _set_pair_route(
        self,
        args,
        pair_strategy,
        interaction_level="",
        interaction_model="",
        predictor_type="classifier",
        graph_predictor="",
    ):
        args.pair_strategy = pair_strategy
        args.interaction_level = interaction_level
        args.interaction_model = interaction_model
        args.predictor_type = predictor_type
        args.graph_predictor = graph_predictor
        normalize_pair_workflow_args(args)

    def _prompt_feature_transformation(self, args, step, allow_back=False):
        options = transformation_options_for_encoder(args.encoder)
        if normalize_task_type(getattr(args, "task_type", "")) in {
            "pep_protein_interaction",
            "pep_hlaii",
            "pep_hlai",
        }:
            options = [option for option in options if option != "lsa"]
        if options == ["none"]:
            args.reduction = "none"
            return step, None
        print(f"Step {step}: Choose feature transformation")
        print("LSA uses uncentered truncated SVD; LDA and PLSA are topic-model transformations.")
        default_reduction = getattr(args, "reduction", "none")
        if default_reduction not in options:
            default_reduction = "none"
        chosen_reduction = self.prompt_choice(
            "Transformation", options, default_reduction, allow_back=allow_back
        )
        if chosen_reduction == "__BACK__":
            return step, "__BACK__"
        args.reduction = chosen_reduction
        if args.reduction in {"lda", "plsa"} and normalize_task_type(
            getattr(args, "task_type", "")
        ) in {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}:
            args.pair_strategy = "two_stream"
            args.interaction_level = "vector"
            args.predictor_type = "classifier"
            args.fusion_mode = "concat"
            if str(getattr(args, "classifier", "") or "").strip().lower() not in {
                "svm",
                "rf",
                "extratrees",
                "histgb",
                "xgb",
                "lgb",
                "logreg",
            }:
                args.classifier = "svm"
        step += 1
        if args.reduction == "lsa":
            args.lsa_dim = self.prompt_int(
                f"Step {step}: LSA dimension", getattr(args, "lsa_dim", 128), 1
            )
            args.lsa_dense_memory_limit_gib = float(
                getattr(
                    args,
                    "lsa_dense_memory_limit_gib",
                    LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                )
            )
            step += 1
        elif args.reduction == "lda":
            args.lda_topics = self.prompt_int(
                f"Step {step}: LDA topic count", getattr(args, "lda_topics", 50), 1
            )
            args.lda_dense_memory_limit_gib = float(
                getattr(
                    args,
                    "lda_dense_memory_limit_gib",
                    LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                )
            )
            step += 1
        elif args.reduction == "plsa":
            args.plsa_topics = self.prompt_int(
                f"Step {step}: PLSA topic count", getattr(args, "plsa_topics", 50), 1
            )
            args.plsa_memory_limit_gib = float(
                getattr(
                    args,
                    "plsa_memory_limit_gib",
                    PLSA_DEFAULT_MEMORY_LIMIT_GIB,
                )
            )
            step += 1
        return step, None

    def _architecture_options_for_encoder(self, encoder):
        architectures = list(SEQUENCE_ARCHITECTURES)
        if str(encoder or "").strip().lower() in CHAIN_GRAPH_COMPATIBLE_ENCODERS:
            architectures.extend(CHAIN_GRAPH_ARCHITECTURES)
        return architectures

    def _prompt_architecture_choice(
        self, args, architectures, default_architecture, allow_back=False, graph_context="sequence"
    ):
        labels = {}
        for architecture in architectures:
            if architecture in CHAIN_GRAPH_ARCHITECTURES:
                family = (
                    "residue chain-graph architecture"
                    if graph_context == "residue"
                    else "sequence chain-graph architecture"
                )
            else:
                family = (
                    "residue sequence architecture"
                    if graph_context == "residue"
                    else "sequence architecture"
                )
            labels[architecture] = f"{architecture} [{family}]"
        default_architecture = (
            default_architecture if default_architecture in architectures else architectures[0]
        )
        chosen = self.prompt_choice(
            "Modeling architecture",
            [labels[architecture] for architecture in architectures],
            labels[default_architecture],
            allow_back=allow_back,
        )
        if chosen == "__BACK__":
            return chosen
        if chosen in architectures:
            return chosen
        return next(architecture for architecture, label in labels.items() if label == chosen)

    def _prompt_sequence_route(self, args, step, allow_back=False):
        print(f"Step {step}: Choose modeling architecture")
        architectures = self._architecture_options_for_encoder(getattr(args, "encoder", ""))
        default_sequence_model = getattr(args, "sequence_model", "") or (
            args.model
            if str(getattr(args, "model", "")).strip().lower() in architectures
            else "cnn"
        )
        chosen_sequence_model = self._prompt_architecture_choice(
            args,
            architectures,
            default_sequence_model,
            allow_back=allow_back,
        )
        if chosen_sequence_model == "__BACK__":
            return step, "__BACK__"
        return self._prompt_training_mode_for_sequence_model(
            args, step + 1, chosen_sequence_model, allow_back=allow_back
        )

    def _prompt_pretrained_single_route(self, args, step, allow_back=False):
        route_labels = ["pooled representation + classifier", "residue modeling"]
        current_architecture = (
            str(getattr(args, "sequence_model", "") or getattr(args, "model", "")).strip().lower()
        )
        explicit_sequence_model = str(getattr(args, "sequence_model", "") or "").strip().lower()
        default_route = (
            route_labels[1]
            if explicit_sequence_model in SEQUENCE_MODELS
            or current_architecture in CHAIN_GRAPH_ARCHITECTURES
            else route_labels[0]
        )
        print(f"Step {step}: Choose predictor route")
        chosen_route = self.prompt_choice(
            "Predictor route",
            route_labels,
            default_route,
            allow_back=allow_back,
        )
        if chosen_route == "__BACK__":
            return step, "__BACK__"
        step += 1
        if chosen_route == route_labels[0]:
            return self._prompt_classifier(args, step, allow_back=allow_back)

        print(f"Step {step}: Choose modeling architecture")
        architecture = self._prompt_architecture_choice(
            args,
            SEQUENCE_MODELS,
            current_architecture if current_architecture in SEQUENCE_MODELS else "cnn",
            allow_back=allow_back,
            graph_context="residue",
        )
        if architecture == "__BACK__":
            return step, "__BACK__"
        step, back = self._prompt_training_mode_for_sequence_model(
            args,
            step + 1,
            architecture,
            allow_back=allow_back,
        )
        if back == "__BACK__":
            return step, back
        default_dim = int(
            getattr(args, "residue_adapter_dim", RESIDUE_ADAPTER_DIM_DEFAULT)
            or RESIDUE_ADAPTER_DIM_DEFAULT
        )
        while True:
            adapter_dim = self.prompt_int(
                f"Step {step}: Residue adapter dimension",
                default_dim,
                RESIDUE_ADAPTER_DIM_MIN,
                allow_back=allow_back,
            )
            if (
                adapter_dim <= RESIDUE_ADAPTER_DIM_MAX
                and adapter_dim % RESIDUE_ADAPTER_DIM_MULTIPLE == 0
            ):
                args.residue_adapter_dim = adapter_dim
                break
            print(
                "Residue adapter dimension must be between "
                f"{RESIDUE_ADAPTER_DIM_MIN} and {RESIDUE_ADAPTER_DIM_MAX}, "
                f"and divisible by {RESIDUE_ADAPTER_DIM_MULTIPLE}."
            )
        return step + 1, None

    def _prompt_training_mode_for_sequence_model(
        self, args, step, sequence_model, allow_back=False
    ):
        print(f"Step {step}: Choose training mode")
        training_labels = [
            "end-to-end prediction",
            "export embeddings for downstream classification",
        ]
        mode_to_label = {
            END_TO_END: training_labels[0],
            EXPORT_EMBEDDINGS: training_labels[1],
        }
        default_training_mode = (
            str(getattr(args, "training_mode", "") or END_TO_END).strip().lower()
        )
        default_label = mode_to_label.get(default_training_mode, training_labels[0])
        chosen_label = self.prompt_choice(
            "Training mode", training_labels, default_label, allow_back=allow_back
        )
        if chosen_label == "__BACK__":
            return step, "__BACK__"
        training_mode = END_TO_END if chosen_label == training_labels[0] else EXPORT_EMBEDDINGS
        step += 1
        classifier = "mlp"
        if training_mode == EXPORT_EMBEDDINGS:
            print(
                f"Export mode first trains {sequence_model} with a temporary neural head "
                "for embedding learning; the downstream classifier is the final predictor."
            )
            print(f"Step {step}: Choose downstream classifier")
            default_classifier = effective_classifier(args)
            if default_classifier not in CLASSICAL_CLASSIFIERS:
                default_classifier = "svm"
            classifier = self.prompt_choice(
                "Downstream classifier",
                CLASSICAL_CLASSIFIERS,
                default_classifier,
                allow_back=allow_back,
            )
            if classifier == "__BACK__":
                return step, "__BACK__"
            step += 1
        self._set_sequence_route(args, sequence_model, training_mode, classifier)
        return step, None

    def _prompt_training_mode_for_interaction_model(
        self, args, step, interaction_model, allow_back=False
    ):
        print(f"Step {step}: Choose training mode")
        training_labels = [
            "end-to-end prediction",
            "export embeddings for downstream classification",
        ]
        mode_to_label = {
            END_TO_END: training_labels[0],
            EXPORT_EMBEDDINGS: training_labels[1],
        }
        default_training_mode = (
            str(getattr(args, "training_mode", "") or END_TO_END).strip().lower()
        )
        default_label = mode_to_label.get(default_training_mode, training_labels[0])
        chosen_label = self.prompt_choice(
            "Training mode", training_labels, default_label, allow_back=allow_back
        )
        if chosen_label == "__BACK__":
            return step, "__BACK__"
        args.training_mode = END_TO_END if chosen_label == training_labels[0] else EXPORT_EMBEDDINGS
        args.classifier = "mlp"
        step += 1
        if args.training_mode == EXPORT_EMBEDDINGS:
            print(
                f"Export mode first trains {interaction_model} with a temporary neural head "
                "for interaction-embedding learning; the downstream classifier is the final predictor."
            )
            print(f"Step {step}: Choose downstream classifier")
            default_classifier = effective_classifier(args)
            if default_classifier not in CLASSICAL_CLASSIFIERS:
                default_classifier = "svm"
            classifier = self.prompt_choice(
                "Downstream classifier",
                CLASSICAL_CLASSIFIERS,
                default_classifier,
                allow_back=allow_back,
            )
            if classifier == "__BACK__":
                return step, "__BACK__"
            args.classifier = classifier
            step += 1
        self._set_pair_route(
            args,
            pair_strategy="two_stream",
            interaction_level=TOKEN_INTERACTION,
            interaction_model=interaction_model,
            predictor_type="classifier",
        )
        return step, None

    def _prompt_pair_strategy(self, args, step, allow_single=True, allow_back=False):
        options = ["single_stream(role-boundary-sequence-join-v1)", "two_stream"]
        if not allow_single:
            options = ["two_stream"]
        default_strategy = normalize_pair_strategy(
            getattr(args, "pair_strategy", "") or "two_stream"
        )
        default_label = options[0]
        if default_strategy == "two_stream":
            default_label = "two_stream"
        print(f"Step {step}: Choose pair strategy")
        chosen = self.prompt_choice("Pair strategy", options, default_label, allow_back=allow_back)
        if chosen == "__BACK__":
            return step, "__BACK__"
        pair_strategy = "single_stream" if chosen.startswith("single_stream") else "two_stream"
        args.pair_strategy = pair_strategy
        return step + 1, None

    def _prompt_two_stream_route(self, args, step, allow_token=True, allow_back=False):
        labels = ["vector-level interaction"]
        if allow_token:
            labels.append("token-level interaction")
        default_level = (
            str(getattr(args, "interaction_level", "") or VECTOR_INTERACTION).strip().lower()
        )
        default_label = (
            "token-level interaction"
            if default_level == TOKEN_INTERACTION and allow_token
            else labels[0]
        )
        print(f"Step {step}: Choose interaction level")
        chosen_level_label = self.prompt_choice(
            "Interaction level", labels, default_label, allow_back=allow_back
        )
        if chosen_level_label == "__BACK__":
            return step, "__BACK__"
        interaction_level = (
            TOKEN_INTERACTION if chosen_level_label.startswith("token") else VECTOR_INTERACTION
        )
        step += 1
        if interaction_level == TOKEN_INTERACTION:
            print(f"Step {step}: Choose interaction architecture")
            default_model = (
                str(getattr(args, "interaction_model", "") or "cross_attn").strip().lower()
            )
            if default_model not in PAIR_INTERACTION_MODELS:
                default_model = "cross_attn"
            interaction_model = self.prompt_choice(
                "Interaction architecture",
                PAIR_INTERACTION_MODELS,
                default_model,
                allow_back=allow_back,
            )
            if interaction_model == "__BACK__":
                return step, "__BACK__"
            return self._prompt_training_mode_for_interaction_model(
                args, step + 1, interaction_model, allow_back=allow_back
            )

        predictor_labels = ["classifier", "sample-graph predictor"]
        default_predictor = str(getattr(args, "predictor_type", "") or "classifier").strip().lower()
        default_label = (
            "sample-graph predictor" if default_predictor == "sample_graph" else "classifier"
        )
        print(f"Step {step}: Choose predictor route")
        chosen_predictor = self.prompt_choice(
            "Predictor route", predictor_labels, default_label, allow_back=allow_back
        )
        if chosen_predictor == "__BACK__":
            return step, "__BACK__"
        step += 1
        if chosen_predictor == "sample-graph predictor":
            print(f"Step {step}: Choose sample-graph architecture")
            default_graph = str(getattr(args, "graph_predictor", "") or "gcn").strip().lower()
            if default_graph not in GRAPH_PREDICTORS:
                default_graph = "gcn"
            graph_predictor = self.prompt_choice(
                "Sample-graph architecture", GRAPH_PREDICTORS, default_graph, allow_back=allow_back
            )
            if graph_predictor == "__BACK__":
                return step, "__BACK__"
            args.graph_predictor = graph_predictor
            self._set_pair_route(
                args,
                pair_strategy="two_stream",
                interaction_level=VECTOR_INTERACTION,
                predictor_type="sample_graph",
                graph_predictor=graph_predictor,
            )
            step += 1
            if graph_predictor == "hgt":
                args.fusion_mode = None
                return step, None
            fusion_options = self._fusion_options_for_model(graph_predictor)
            fusion_choice = self.prompt_choice(
                f"Step {step}: Choose fusion",
                fusion_options,
                self._fusion_default(args, fusion_options),
                allow_back=allow_back,
            )
            if fusion_choice == "__BACK__":
                return step, "__BACK__"
            self._apply_fusion_choice(args, fusion_choice)
            return step + 1, None

        step, classifier_back = self._prompt_classifier(args, step, allow_back=allow_back)
        if classifier_back == "__BACK__":
            return step, "__BACK__"
        self._set_pair_route(
            args,
            pair_strategy="two_stream",
            interaction_level=VECTOR_INTERACTION,
            predictor_type="classifier",
        )
        fusion_options = self._fusion_options_for_model(args.classifier)
        fusion_choice = self.prompt_choice(
            f"Step {step}: Choose fusion",
            fusion_options,
            self._fusion_default(args, fusion_options),
            allow_back=allow_back,
        )
        if fusion_choice == "__BACK__":
            return step, "__BACK__"
        self._apply_fusion_choice(args, fusion_choice)
        return step + 1, None

    def _prompt_classifier(
        self, args, step, title="Choose classifier", options=None, allow_back=False
    ):
        print(f"Step {step}: {title}")
        options = options or CLASSIFIERS
        default_classifier = effective_classifier(args)
        if default_classifier not in options:
            default_classifier = options[0]
        classifier = self.prompt_choice(
            "Classifier", options, default_classifier, allow_back=allow_back
        )
        if classifier == "__BACK__":
            return step, "__BACK__"
        self._set_classifier(args, classifier)
        return step + 1, None

    def _is_trainable_dl_route(self, args):
        return args.model in TRAINABLE_DL_MODEL_NAMES

    def _needs_ml_settings(self, args):
        return effective_classifier(args) in CLASSICAL_CLASSIFIERS

    def _regularization_relevant(self, args):
        return self._is_trainable_dl_route(args) or effective_classifier(args) in {
            "svm",
            "logreg",
            "histgb",
            "xgb",
            "lgb",
        }

    def _prediction_head_supported(self, args):
        return prediction_head_applies(args)

    def _prompt_prediction_head(self, args, step, allow_back=False):
        if not self._prediction_head_supported(args):
            return step, None
        route_default = default_prediction_head_hidden_dims(args)
        current = getattr(args, "head_hidden_dims", None)
        if current == []:
            default_preset = "linear"
        elif current not in [None, route_default]:
            default_preset = "custom"
        else:
            default_preset = "default"
        preset = self.prompt_choice(
            f"Step {step}: Prediction head",
            ["default", "linear", "custom"],
            default_preset,
            allow_back=allow_back,
        )
        if preset == "__BACK__":
            return step, "__BACK__"
        step += 1
        if preset == "linear":
            args.head_hidden_dims = []
            args.head_activation = None
            args.head_dropout = None
            return step, None
        if preset == "default":
            args.head_hidden_dims = route_default
        else:
            default_text = json.dumps(current if current not in [None, []] else route_default)
            while True:
                value = self.prompt_text(
                    f"Step {step}: Hidden dimensions as JSON",
                    default_text,
                    allow_back=allow_back,
                )
                if value == "__BACK__":
                    return step, "__BACK__"
                try:
                    args.head_hidden_dims = parse_prediction_head_hidden_dims(value)
                except ValueError as exc:
                    print(exc)
                    continue
                if not args.head_hidden_dims:
                    print("Use the linear preset for an empty hidden-dimension list.")
                    continue
                break
            step += 1
        args.head_activation = self.prompt_choice(
            f"Step {step}: Head activation",
            ["relu", "gelu", "silu"],
            str(getattr(args, "head_activation", "relu") or "relu"),
            allow_back=allow_back,
        )
        if args.head_activation == "__BACK__":
            return step, "__BACK__"
        step += 1
        args.head_dropout = self.prompt_float(
            f"Step {step}: Head dropout",
            float(getattr(args, "head_dropout", getattr(args, "dropout", 0.0)) or 0.0),
            0.0,
            allow_back=allow_back,
        )
        if args.head_dropout == "__BACK__":
            return step, "__BACK__"
        return step + 1, None

    def _semantic_auxiliary_supported(self, args):
        task = normalize_task_type(getattr(args, "task_type", ""))
        model = str(getattr(args, "model", "")).strip().lower()
        training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
        if (
            training_mode == "export_embeddings"
            or effective_classifier(args) in CLASSICAL_CLASSIFIERS
        ):
            return False
        if normalize_task_type(task) == "pep_function":
            return model in (DL_SEQUENCE_MODEL_NAMES + ["mlp"])
        if normalize_task_type(task) in ["pep_protein_interaction", "pep_hlaii", "pep_hlai"]:
            return (
                model in PAIR_SEMANTIC_MODEL_NAMES
                and str(getattr(args, "predictor_type", "classifier") or "classifier")
                .strip()
                .lower()
                != "sample_graph"
            )
        return False

    def _semantic_auxiliary_enabled(self, args):
        return bool(getattr(args, "enable_semantic_similarity", False)) or bool(
            getattr(args, "enable_interaction_semantic_similarity", False)
        )

    def _set_semantic_auxiliary(self, args, enabled):
        task = normalize_task_type(getattr(args, "task_type", ""))
        enabled = bool(enabled)
        args.enable_semantic_similarity = enabled
        args.semantic_similarity_weight = float(getattr(args, "semantic_similarity_weight", 0.1))
        args.semantic_zero_weight = float(getattr(args, "semantic_zero_weight", 0.25))
        if normalize_task_type(task) != "pep_function":
            args.enable_interaction_semantic_similarity = None
            args.interaction_semantic_mode = ""
            args.interaction_semantic_similarity_weight = None

    def _prompt_semantic_auxiliary(self, args, step, allow_back=False):
        if not self._semantic_auxiliary_supported(args):
            self._set_semantic_auxiliary(args, False)
            return step, None
        default_semantic = "yes" if self._semantic_auxiliary_enabled(args) else "no"
        semantic_choice = self.prompt_choice(
            f"Step {step}: Enable semantic similarity (auxiliary loss)",
            ["yes", "no"],
            default_semantic,
            allow_back=allow_back,
        )
        if semantic_choice == "__BACK__":
            return step, "__BACK__"
        self._set_semantic_auxiliary(args, semantic_choice == "yes")
        return step + 1, None

    def prompt_choice(self, title, options, default_value, allow_back=False):
        if default_value in options:
            default_idx = options.index(default_value) + 1
        else:
            default_idx = 1
        while True:
            print(f"{title} [1-{len(options)}] (default {default_idx}):")
            for i, opt in enumerate(options, start=1):
                print(f"{i}. {opt}")
            raw = input("Choice: ").strip()
            if raw == "__BACK__":
                print("Back to previous step is disabled. Please choose a valid value.")
                continue
            if raw == "":
                return options[default_idx - 1]
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1]
            if raw in options:
                return raw
            print("Invalid input. Please retry.")

    def prompt_int(self, title, default_value, min_value=None, allow_back=False, max_value=None):
        while True:
            raw = input(f"{title} (default {default_value}): ").strip()
            if raw == "__BACK__":
                print("Back to previous step is disabled. Please enter a valid integer.")
                continue
            if raw == "":
                val = default_value
            else:
                try:
                    val = int(raw)
                except ValueError:
                    print("Please enter an integer.")
                    continue
            if min_value is not None and val < min_value:
                print(f"Please enter an integer >= {min_value}.")
                continue
            if max_value is not None and val > max_value:
                print(f"Please enter an integer <= {max_value}.")
                continue
            return val

    def prompt_float(self, title, default_value, min_value=None, allow_back=False):
        while True:
            raw = input(f"{title} (default {default_value}): ").strip()
            if raw == "__BACK__":
                print("Back to previous step is disabled. Please enter a valid number.")
                continue
            if raw == "":
                return default_value
            try:
                val = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
            if not math.isfinite(val):
                print("Please enter a finite number.")
                continue
            if min_value is not None and val < min_value:
                print(f"Please enter a number >= {min_value}.")
                continue
            return val

    def prompt_text(self, title, default_value, allow_back=False):
        while True:
            raw = input(f"{title} (default {default_value}): ").strip()
            if raw == "__BACK__":
                print("Back to previous step is disabled. Please enter a valid text value.")
                continue
            return raw if raw else default_value

    def run(self, args):
        step = 1
        print("Entering interactive mode")
        print(f"Step {step}: Choose task")
        task_options = ["pep_function", "pep_protein_interaction", "pep_hlaii", "pep_hlai"]
        default_task = normalize_task_type(getattr(args, "task_type", "pep_function"))
        if normalize_task_type(default_task) == "pep_function_multilabel":
            default_task = "pep_function"
        if normalize_task_type(default_task) not in task_options:
            default_task = task_options[0]
        args.task_type = self.prompt_choice("Task", task_options, default_task)
        if int(getattr(args, "epochs", 30)) == 30:
            if normalize_task_type(args.task_type) == "pep_hlaii":
                args.epochs = 10
            elif normalize_task_type(args.task_type) == "pep_hlai":
                args.epochs = 10
        task_root = self._task_root(args.task_type)
        step += 1

        if normalize_task_type(args.task_type) in [
            "pep_protein_interaction",
            "pep_hlaii",
            "pep_hlai",
        ]:
            print(f"Step {step}: Choose feature extraction")
            default_mode = (
                "pretrained model"
                if normalize_tokenizer_name(getattr(args, "tokenizer", ""))
                in ["esm2", "protbert", "prott5", "esmc", "esm3"]
                else "classic pipeline"
            )
            feature_mode = self.prompt_choice(
                "Feature extraction", ["pretrained model", "classic pipeline"], default_mode
            )
            step += 1

            if feature_mode == "pretrained model":
                print(f"Step {step}: Choose pretrained model")
                tokenizer_options = ["esm2", "protbert", "prott5", "esmc", "esm3"]
                default_tok = normalize_tokenizer_name(getattr(args, "tokenizer", "esm2"))
                if default_tok not in tokenizer_options:
                    default_tok = "esm2"
                args.tokenizer = self.prompt_choice(
                    "Pretrained model", tokenizer_options, default_tok
                )
                args.encoder = args.tokenizer
                print(f"Encoder: {args.encoder} (coupled to pretrained tokenizer)")
                step += 1
            else:
                print(f"Step {step}: Choose tokenizer")
                tokenizer_options = [
                    "kmer",
                    "dr",
                    "top-n-gram",
                    "dt",
                    "bpe",
                    "wordpiece",
                    "unigram",
                ]
                default_tok = getattr(args, "tokenizer", "kmer")
                if default_tok not in tokenizer_options:
                    default_tok = "kmer"
                args.tokenizer = self.prompt_choice("Tokenizer", tokenizer_options, default_tok)
                step += 1

                print(f"Step {step}: Choose encoder")
                if args.tokenizer in ["bpe", "wordpiece", "unigram"]:
                    encoder_options = ["index"]
                    default_enc = "index"
                elif args.tokenizer == "dr":
                    encoder_options = [
                        "index",
                        "onehot",
                        "bow",
                        "tfidf",
                        "textrank",
                        "word2vec",
                        "glove",
                        "fasttext",
                    ]
                    default_enc = getattr(args, "encoder", "tfidf")
                elif args.tokenizer in ["top-n-gram", "dt"]:
                    encoder_options = ["bow", "tfidf", "textrank"]
                    default_enc = getattr(args, "encoder", "tfidf")
                else:
                    encoder_options = [
                        "index",
                        "onehot",
                        "bow",
                        "tfidf",
                        "textrank",
                        "ac",
                        "cc",
                        "acc",
                        "pdt",
                        "blosum62",
                        "pssm",
                        "pssm_blosum62_hybrid",
                        "psfm",
                        "psfm_blosum62_hybrid",
                        "ac_pssm",
                        "cc_pssm",
                        "acc_pssm",
                        "ac_pssm_blosum62_hybrid",
                        "cc_pssm_blosum62_hybrid",
                        "acc_pssm_blosum62_hybrid",
                        "word2vec",
                        "glove",
                        "fasttext",
                    ]
                    default_enc = getattr(args, "encoder", "tfidf")
                if default_enc not in encoder_options:
                    default_enc = encoder_options[0]
                chosen_encoder = self.prompt_choice("Encoder", encoder_options, default_enc)
                args.encoder = chosen_encoder
                step += 1
                step, _ = self._prompt_feature_transformation(args, step)

                if args.tokenizer in {"top-n-gram", "dt"}:
                    default_frequency_index = (
                        getattr(
                            args,
                            "frequency_profile_index_path",
                            "",
                        )
                        or f"{task_root}/frequency_profile_cache/ppi/index.json"
                    )
                    args.frequency_profile_index_path = self.prompt_text(
                        f"Step {step}: target-frequency profile index path",
                        default_frequency_index,
                        allow_back=False,
                    )
                    args.profile_on_missing = TOP_N_GRAM_FALLBACK_POLICY
                    step += 1
                elif args.encoder in [
                    "pssm",
                    "pssm_blosum62_hybrid",
                    "ac_pssm",
                    "cc_pssm",
                    "acc_pssm",
                ]:
                    default_cache = (
                        "pssm_hybrid_cache"
                        if args.encoder == "pssm_blosum62_hybrid"
                        else "pssm_cache"
                    )
                    default_pssm_index = (
                        getattr(args, "pssm_index_path", "")
                        or f"{task_root}/{default_cache}/ppi/index.json"
                    )
                    args.pssm_index_path = self.prompt_text(
                        f"Step {step}: PSSM index path", default_pssm_index, allow_back=False
                    )
                    args.profile_on_missing = (
                        "blosum62_hybrid" if args.encoder == "pssm_blosum62_hybrid" else "error"
                    )
                    step += 1
                elif args.encoder in HYBRID_PSSM_COVARIANCE_ENCODERS:
                    index_key = f"{args.encoder}_index_path"
                    default_index = (
                        getattr(args, index_key, "")
                        or f"{task_root}/{args.encoder}_cache/ppi/index.json"
                    )
                    setattr(
                        args,
                        index_key,
                        self.prompt_text(
                            f"Step {step}: {args.encoder} index path",
                            default_index,
                            allow_back=False,
                        ),
                    )
                    args.profile_on_missing = "blosum62_hybrid"
                    step += 1
                if args.tokenizer == "top-n-gram":
                    maximum_top_n = 3
                    while True:
                        top_n = self.prompt_int(
                            f"Step {step}: top n",
                            getattr(args, "top_ngram_n", 2),
                            1,
                        )
                        if top_n <= maximum_top_n:
                            args.top_ngram_n = top_n
                            break
                        print(f"top_ngram_n must be in [1,{maximum_top_n}], please reselect.")
                    step += 1
                if args.tokenizer == "dt":
                    args.dt_max_distance = self.prompt_int(
                        f"Step {step}: DT max distance",
                        getattr(args, "dt_max_distance", 5),
                        0,
                        max_value=DT_MAX_DISTANCE,
                    )
                    step += 1
                if args.encoder in {"psfm", "psfm_blosum62_hybrid"}:
                    default_psfm_index = (
                        getattr(args, "psfm_index_path", "")
                        or f"{task_root}/{'psfm_hybrid_cache' if args.encoder == 'psfm_blosum62_hybrid' else 'psfm_cache'}/ppi/index.json"
                    )
                    args.psfm_index_path = self.prompt_text(
                        f"Step {step}: PSFM index path", default_psfm_index, allow_back=False
                    )
                    args.profile_on_missing = (
                        "blosum62_hybrid" if args.encoder == "psfm_blosum62_hybrid" else "error"
                    )
                    step += 1
                if args.encoder == "ac_pssm":
                    args.ac_pssm_lag = self.prompt_int(
                        f"Step {step}: AC-PSSM lag", getattr(args, "ac_pssm_lag", 10), 1
                    )
                    step += 1
                if args.encoder == "cc_pssm":
                    args.cc_pssm_lag = self.prompt_int(
                        f"Step {step}: CC-PSSM lag", getattr(args, "cc_pssm_lag", 10), 1
                    )
                    step += 1
                if args.encoder == "acc_pssm":
                    args.acc_pssm_lag = self.prompt_int(
                        f"Step {step}: ACC-PSSM lag", getattr(args, "acc_pssm_lag", 10), 1
                    )
                    step += 1
                if args.encoder in HYBRID_PSSM_COVARIANCE_ENCODERS:
                    lag_key = pssm_covariance_lag_key(args.encoder)
                    setattr(
                        args,
                        lag_key,
                        self.prompt_int(
                            f"Step {step}: {args.encoder} lag",
                            getattr(args, lag_key, 10),
                            1,
                        ),
                    )
                    step += 1

            enc_name = str(getattr(args, "encoder", "index")).lower()
            seq_encoder_set = {
                "index",
                "onehot",
                "blosum62",
                "pssm",
                "pssm_blosum62_hybrid",
                "psfm",
                "psfm_blosum62_hybrid",
                "word2vec",
                "glove",
                "fasttext",
            }
            allow_single = feature_mode == "classic pipeline" and enc_name in seq_encoder_set
            allow_token = feature_mode == "pretrained model" or enc_name in seq_encoder_set
            step, _ = self._prompt_pair_strategy(args, step, allow_single=allow_single)
            if args.pair_strategy == "single_stream":
                step, _ = self._prompt_sequence_route(args, step)
                self._set_pair_route(args, pair_strategy="single_stream")
            else:
                step, _ = self._prompt_two_stream_route(args, step, allow_token=allow_token)
            step, semantic_back = self._prompt_semantic_auxiliary(args, step)
            if semantic_back == "__BACK__":
                print("Returning to previous step")
                return self.run(args)

            args.max_length = self.prompt_int(
                f"Step {step}: max length", getattr(args, "max_length", 512), 1
            )
            step += 1
            args.batch_size = self.prompt_int(
                f"Step {step}: batch_size", getattr(args, "batch_size", 16), 1
            )
            step += 1

            if args.model in (
                DL_SEQUENCE_MODEL_NAMES + ["mlp", "gcn", "gat", "hgt", "sage", "graph_transformer"]
            ):
                args.dl_scaler = self.prompt_choice(
                    "DL feature scaling",
                    ["none", "standard", "minmax"],
                    getattr(args, "dl_scaler", "standard"),
                )
                step += 1
                args.lr = self.prompt_float(
                    f"Step {step}: learning rate", getattr(args, "lr", 1e-3), 0.0
                )
                step += 1
                args.epochs = self.prompt_int(
                    f"Step {step}: epochs", getattr(args, "epochs", 20), 1
                )
                step += 1
                args.dropout = self.prompt_float(
                    f"Step {step}: dropout", getattr(args, "dropout", 0.2), 0.0
                )
                step += 1
                args.weight_decay = self.prompt_float(
                    f"Step {step}: weight decay", getattr(args, "weight_decay", 1e-2), 0.0
                )
                step += 1
                if getattr(args, "predictor_type", "") == "sample_graph" and args.model in [
                    "gcn",
                    "gat",
                    "hgt",
                    "sage",
                    "graph_transformer",
                ]:
                    graph_name = (
                        "GCN"
                        if args.model == "gcn"
                        else (
                            "GAT"
                            if args.model == "gat"
                            else (
                                "HGT"
                                if args.model == "hgt"
                                else ("GraphSAGE" if args.model == "sage" else "GraphTransformer")
                            )
                        )
                    )
                    print(f"{graph_name} selected: sample-graph")
                    args.gnn_k = self.prompt_int(
                        f"Step {step}: {graph_name} kNN k", int(getattr(args, "gnn_k", 8)), 1
                    )
                    step += 1
                    args.gnn_hidden_dim = self.prompt_int(
                        f"Step {step}: {graph_name} hidden dim",
                        int(getattr(args, "gnn_hidden_dim", 256)),
                        8,
                    )
                    step += 1
                args.early_stop_patience = self.prompt_int(
                    f"Step {step}: early stop patience", getattr(args, "early_stop_patience", 5), 0
                )
                step += 1
                args.early_stop_min_delta = self.prompt_float(
                    f"Step {step}: early stop min delta",
                    getattr(args, "early_stop_min_delta", 0.0),
                    0.0,
                )
                step += 1
            step, _ = self._prompt_prediction_head(args, step)
            if self._needs_ml_settings(args):
                args.ml_scaler = self.prompt_choice(
                    "ML scaler",
                    ["none", "standard", "minmax"],
                    getattr(args, "ml_scaler", "standard"),
                )
                step += 1

            default_aupr = "yes" if getattr(args, "report_aupr", True) else "no"
            use_aupr = self.prompt_choice(f"Step {step}: Report AUPR", ["yes", "no"], default_aupr)
            args.report_aupr = use_aupr == "yes"
            step += 1

            args.task_root = self.prompt_text(
                f"Step {step}: Task root", getattr(args, "task_root", task_root), allow_back=False
            )
            step += 1
            args.train_csv = self.prompt_text(
                f"Step {step}: Train CSV (blank=default)",
                getattr(args, "train_csv", ""),
                allow_back=False,
            )
            step += 1
            args.val_csv = self.prompt_text(
                f"Step {step}: Val CSV (blank=split train)",
                getattr(args, "val_csv", ""),
                allow_back=False,
            )
            step += 1
            args.test_csv = self.prompt_text(
                f"Step {step}: Test CSV (blank=default)",
                getattr(args, "test_csv", ""),
                allow_back=False,
            )
            step += 1
            args.feature_dir = self.prompt_text(
                f"Step {step}: Feature directory (blank=default)",
                getattr(args, "feature_dir", ""),
                allow_back=False,
            )
            step += 1
            default_cache = "yes" if getattr(args, "cache_features", True) else "no"
            cache_choice = self.prompt_choice("Cache features", ["yes", "no"], default_cache)
            args.cache_features = cache_choice == "yes"

            print("Configuration complete")
            print(f"task_type={args.task_type}")
            print(f"tokenizer={args.tokenizer}")
            print(f"encoder={getattr(args, 'encoder', '')}")
            print(f"model={args.model}")
            print(f"pair_strategy={public_pair_strategy(getattr(args, 'pair_strategy', ''))}")
            if getattr(args, "interaction_level", ""):
                print(f"interaction_level={args.interaction_level}")
            if getattr(args, "interaction_model", ""):
                print(f"interaction_model={args.interaction_model}")
            print(f"predictor_type={getattr(args, 'predictor_type', 'classifier')}")
            if getattr(args, "graph_predictor", ""):
                print(f"graph_predictor={args.graph_predictor}")
            if getattr(args, "sequence_model", ""):
                print(f"sequence_model={args.sequence_model}")
                print(f"training_mode={args.training_mode}")
                print(f"classifier={args.classifier}")
            if getattr(args, "head_hidden_dims", None) is not None:
                print(f"head_hidden_dims={json.dumps(args.head_hidden_dims)}")
                if args.head_hidden_dims:
                    print(f"head_activation={args.head_activation}")
                    print(f"head_dropout={args.head_dropout}")
            if args.model == "hgt":
                print("input_adapter=typed_role_projection")
            elif args.model in [
                "mlp",
                "svm",
                "rf",
                "extratrees",
                "histgb",
                "xgb",
                "lgb",
                "logreg",
                "gcn",
                "gat",
                "sage",
                "graph_transformer",
            ]:
                print(f"fusion_mode={getattr(args, 'fusion_mode', 'concat_all')}")
            print(f"enable_semantic_similarity={self._semantic_auxiliary_enabled(args)}")
            if self._semantic_auxiliary_enabled(args):
                print("semantic_similarity_type=auxiliary_loss")
            if getattr(args, "predictor_type", "") == "sample_graph" and args.model in [
                "gcn",
                "gat",
                "hgt",
                "sage",
                "graph_transformer",
            ]:
                print(f"graph_model={args.model}(sample_graph)")
                print(f"gnn_k={getattr(args, 'gnn_k', 8)}")
                print(f"gnn_hidden_dim={getattr(args, 'gnn_hidden_dim', 256)}")
            print(f"report_aupr={getattr(args, 'report_aupr', True)}")
            print(f"task_root={args.task_root}")
            if getattr(args, "train_csv", ""):
                print(f"train_csv={args.train_csv}")
            if getattr(args, "val_csv", ""):
                print(f"val_csv={args.val_csv}")
            if getattr(args, "test_csv", ""):
                print(f"test_csv={args.test_csv}")
            if getattr(args, "feature_dir", ""):
                print(f"feature_dir={args.feature_dir}")
            print(f"cache_features={getattr(args, 'cache_features', True)}")
            return args

        pretrained_tokenizers = ["esm2", "protbert", "prott5", "esmc", "esm3"]
        classic_tokenizers = ["kmer", "dr", "top-n-gram", "dt", "bpe", "wordpiece", "unigram"]
        print(f"Step {step}: Choose feature extraction")
        args.tokenizer = normalize_tokenizer_name(getattr(args, "tokenizer", ""))
        default_mode = (
            "pretrained model"
            if args.tokenizer in ["esm2", "protbert", "prott5", "esmc", "esm3"]
            else "classic pipeline"
        )
        feature_mode = self.prompt_choice(
            "Feature extraction", ["pretrained model", "classic pipeline"], default_mode
        )
        default_tokenizer = args.tokenizer
        if feature_mode == "pretrained model":
            print("Step 2.1: Choose pretrained model")
            if default_tokenizer not in pretrained_tokenizers:
                default_tokenizer = pretrained_tokenizers[0]
            args.tokenizer = self.prompt_choice(
                "Pretrained model", pretrained_tokenizers, default_tokenizer
            )
        else:
            print("Step 2.1: Choose tokenizer")
            if default_tokenizer not in classic_tokenizers:
                default_tokenizer = classic_tokenizers[0]
            args.tokenizer = self.prompt_choice("Tokenizer", classic_tokenizers, default_tokenizer)
        args.tokenizer = normalize_tokenizer_name(args.tokenizer)
        step += 1
        if args.tokenizer in ["esm2", "protbert", "prott5", "esmc", "esm3"]:
            args.encoder = args.tokenizer
            print(f"Encoder: {args.encoder} (coupled to pretrained tokenizer)")
            step, route_back = self._prompt_pretrained_single_route(args, step, allow_back=True)
            if route_back == "__BACK__":
                print("Returning to previous step")
                return self.run(args)
            max_length = self.prompt_int(
                f"Step {step}: max length", args.max_length, 1, allow_back=True
            )
            if max_length == "__BACK__":
                print("Returning to previous step")
                return self.run(args)
            args.max_length = max_length
            step += 1
            if args.tokenizer == "protbert":
                default_dir = (
                    args.protbert_feature_dir
                    if hasattr(args, "protbert_feature_dir") and args.protbert_feature_dir
                    else self._default_pretrained_feature_dir(args, task_root, "protbert")
                )
                protbert_dir = self.prompt_text(
                    f"Step {step}: ProtBERT feature directory", default_dir, allow_back=True
                )
                if protbert_dir == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.protbert_feature_dir = protbert_dir
                step += 1
                default_model_name = (
                    args.protbert_model
                    if hasattr(args, "protbert_model") and args.protbert_model
                    else "Rostlab/prot_bert"
                )
                protbert_model = self.prompt_text(
                    f"Step {step}: ProtBERT model name", default_model_name, allow_back=True
                )
                if protbert_model == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.protbert_model = protbert_model
                step += 1
            if args.tokenizer == "prott5":
                default_dir = (
                    args.prott5_feature_dir
                    if hasattr(args, "prott5_feature_dir") and args.prott5_feature_dir
                    else self._default_pretrained_feature_dir(args, task_root, "prott5")
                )
                prott5_dir = self.prompt_text(
                    f"Step {step}: ProtT5 feature directory", default_dir, allow_back=True
                )
                if prott5_dir == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.prott5_feature_dir = prott5_dir
                step += 1
                default_model_name = (
                    args.prott5_model
                    if hasattr(args, "prott5_model") and args.prott5_model
                    else "Rostlab/prot_t5_xl_uniref50"
                )
                prott5_model = self.prompt_text(
                    f"Step {step}: ProtT5 model name", default_model_name, allow_back=True
                )
                if prott5_model == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.prott5_model = prott5_model
                step += 1
            if args.tokenizer == "esmc":
                default_dir = (
                    args.esmc_feature_dir
                    if hasattr(args, "esmc_feature_dir") and args.esmc_feature_dir
                    else self._default_pretrained_feature_dir(args, task_root, "esmc")
                )
                esmc_dir = self.prompt_text(
                    f"Step {step}: ESMC feature directory", default_dir, allow_back=True
                )
                if esmc_dir == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.esmc_feature_dir = esmc_dir
                step += 1
                esmc_options = [str(option["id"]) for option in official_esmc_model_options()]
                default_model_name = (
                    args.esmc_model
                    if hasattr(args, "esmc_model") and args.esmc_model
                    else PRETRAINED_MODEL_DEFAULTS["esmc"]
                )
                if default_model_name not in esmc_options:
                    default_model_name = PRETRAINED_MODEL_DEFAULTS["esmc"]
                esmc_model = self.prompt_choice(
                    f"Step {step}: ESM-C checkpoint",
                    esmc_options,
                    default_model_name,
                    allow_back=True,
                )
                if esmc_model == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.esmc_model = esmc_model
                step += 1
            if args.tokenizer == "esm3":
                default_dir = (
                    args.esm3_feature_dir
                    if hasattr(args, "esm3_feature_dir") and args.esm3_feature_dir
                    else self._default_pretrained_feature_dir(args, task_root, "esm3")
                )
                esm3_dir = self.prompt_text(
                    f"Step {step}: ESM3 feature directory", default_dir, allow_back=True
                )
                if esm3_dir == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.esm3_feature_dir = esm3_dir
                step += 1
                default_model_name = (
                    args.esm3_model
                    if hasattr(args, "esm3_model") and args.esm3_model
                    else "EvolutionaryScale/esm3-sm-open-v1"
                )
                esm3_model = self.prompt_text(
                    f"Step {step}: ESM3 model name", default_model_name, allow_back=True
                )
                if esm3_model == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.esm3_model = esm3_model
                step += 1

            if args.model == "mlp":
                args.batch_size = self.prompt_int(f"Step {step}: batch_size", args.batch_size, 1)
                step += 1
                args.lr = self.prompt_float(f"Step {step}: learning rate", args.lr, 0.0)
                step += 1
                args.epochs = self.prompt_int(f"Step {step}: epochs", args.epochs, 1)
                step += 1
                args.dl_scaler = self.prompt_choice(
                    "DL feature scaling", ["none", "standard"], args.dl_scaler
                )
                step += 1
                step, semantic_back = self._prompt_semantic_auxiliary(args, step)
                if semantic_back == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                default_enable_reg = "yes" if getattr(args, "enable_regularization", True) else "no"
                enable_reg = self.prompt_choice(
                    "Enable regularization", ["yes", "no"], default_enable_reg
                )
                args.enable_regularization = enable_reg == "yes"
                step += 1
                if args.enable_regularization:
                    args.dropout = self.prompt_float(f"Step {step}: dropout", args.dropout, 0.0)
                    step += 1
                    args.weight_decay = self.prompt_float(
                        f"Step {step}: weight decay", args.weight_decay, 0.0
                    )
                    step += 1
                    args.grad_clip_norm = self.prompt_float(
                        f"Step {step}: grad clip norm (0=off)", args.grad_clip_norm, 0.0
                    )
                    step += 1
                    args.early_stop_patience = self.prompt_int(
                        f"Step {step}: early stop patience (0=off)", args.early_stop_patience, 0
                    )
                    step += 1
                    args.early_stop_min_delta = self.prompt_float(
                        f"Step {step}: early stop min delta", args.early_stop_min_delta, 0.0
                    )
                    step += 1
                else:
                    args.dropout = 0.0
                    args.weight_decay = 0.0
                    args.grad_clip_norm = 0.0
                    args.early_stop_patience = 0
                    args.early_stop_min_delta = 0.0
            else:
                args.dl_scaler = "none"
                args.enable_semantic_similarity = False
                if self._regularization_relevant(args):
                    default_enable_reg = (
                        "yes" if getattr(args, "enable_regularization", True) else "no"
                    )
                    enable_reg = self.prompt_choice(
                        "Enable regularization", ["yes", "no"], default_enable_reg
                    )
                    args.enable_regularization = enable_reg == "yes"
                    step += 1
                else:
                    args.enable_regularization = False
                args.ml_scaler = self.prompt_choice(
                    "ML scaler", ["none", "standard", "minmax"], args.ml_scaler
                )
                step += 1
                args.ml_reduction = self.prompt_choice(
                    "ML reduction", ["none", "pca", "kpca", "tsvd"], args.ml_reduction
                )
                step += 1
                if args.ml_reduction != "none":
                    args.ml_reduction_dim = self.prompt_int(
                        f"Step {step}: ML reduction dimension", args.ml_reduction_dim, 1
                    )
                    step += 1
                if args.ml_reduction == "kpca":
                    args.kpca_kernel = self.prompt_choice(
                        "KPCA kernel", ["rbf", "linear", "poly", "cosine"], args.kpca_kernel
                    )
                    step += 1
                    if args.kpca_kernel in {"rbf", "poly"}:
                        args.kpca_gamma = self.prompt_float(
                            f"Step {step}: KPCA gamma",
                            getattr(args, "kpca_gamma", None) or 0.1,
                            0.0,
                        )
                        step += 1
                    else:
                        args.kpca_gamma = None
                if args.model == "svm":
                    args.svm_calibration = self.prompt_choice(
                        "SVM calibration",
                        ["none", "sigmoid"],
                        getattr(args, "svm_calibration", "none"),
                    )
                    args.svm_calibration_cv_folds = 3
                    step += 1
                    if args.enable_regularization:
                        args.svm_class_weight = self.prompt_choice(
                            "SVM class weight", ["balanced", "none"], args.svm_class_weight
                        )
                        step += 1
                    else:
                        args.svm_class_weight = "none"
                if args.model == "logreg":
                    if args.enable_regularization:
                        args.logreg_class_weight = self.prompt_choice(
                            "LogReg class weight", ["balanced", "none"], args.logreg_class_weight
                        )
                        step += 1
                    else:
                        args.logreg_class_weight = "none"
                    args.logreg_c = self.prompt_float(
                        f"Step {step}: LogReg C", args.logreg_c, 1e-12
                    )
                    step += 1
                    args.logreg_max_iter = self.prompt_int(
                        f"Step {step}: LogReg max_iter", args.logreg_max_iter, 1
                    )
                    step += 1
                if args.model == "rf":
                    args.rf_estimators = self.prompt_int(
                        f"Step {step}: RF estimators", args.rf_estimators, 1
                    )
                    step += 1
                    args.rf_max_depth = self.prompt_int(
                        f"Step {step}: RF max depth (0 = unlimited)", args.rf_max_depth, 0
                    )
                    step += 1
                    args.rf_min_samples_leaf = self.prompt_int(
                        f"Step {step}: RF min samples per leaf", args.rf_min_samples_leaf, 1
                    )
                    step += 1
                if args.model == "extratrees":
                    args.et_estimators = self.prompt_int(
                        f"Step {step}: ExtraTrees estimators", args.et_estimators, 1
                    )
                    step += 1
                    args.et_max_depth = self.prompt_int(
                        f"Step {step}: ExtraTrees max depth (0 = unlimited)", args.et_max_depth, 0
                    )
                    step += 1
                    args.et_min_samples_leaf = self.prompt_int(
                        f"Step {step}: ExtraTrees min samples per leaf", args.et_min_samples_leaf, 1
                    )
                    step += 1
                if args.model == "histgb":
                    args.histgb_max_iter = self.prompt_int(
                        f"Step {step}: HistGB max_iter", args.histgb_max_iter, 1
                    )
                    step += 1
                    while True:
                        args.histgb_max_leaf_nodes = self.prompt_int(
                            f"Step {step}: HistGB max leaf nodes (0 = unlimited)",
                            args.histgb_max_leaf_nodes,
                            0,
                        )
                        if args.histgb_max_leaf_nodes == 0 or args.histgb_max_leaf_nodes >= 2:
                            break
                        print("HistGB max leaf nodes must be 0 (unlimited) or at least 2.")
                    step += 1
                    args.histgb_lr = self.prompt_float(
                        f"Step {step}: HistGB learning rate", args.histgb_lr, 1e-12
                    )
                    step += 1
                    args.histgb_l2_regularization = self.prompt_float(
                        f"Step {step}: HistGB L2 regularization", args.histgb_l2_regularization, 0.0
                    )
                    step += 1
                if args.model == "xgb":
                    args.xgb_estimators = self.prompt_int(
                        f"Step {step}: XGBoost estimators", args.xgb_estimators, 1
                    )
                    step += 1
                    args.xgb_max_depth = self.prompt_int(
                        f"Step {step}: XGBoost max depth (0 = unlimited)", args.xgb_max_depth, 0
                    )
                    step += 1
                    args.xgb_lr = self.prompt_float(
                        f"Step {step}: XGBoost learning rate", args.xgb_lr, 0.0
                    )
                    step += 1
                    if args.enable_regularization:
                        args.xgb_subsample = self.prompt_float(
                            f"Step {step}: XGBoost subsample", args.xgb_subsample, 0.0
                        )
                        step += 1
                        args.xgb_colsample = self.prompt_float(
                            f"Step {step}: XGBoost colsample_bytree", args.xgb_colsample, 0.0
                        )
                        step += 1
                        args.xgb_reg_alpha = self.prompt_float(
                            f"Step {step}: XGBoost reg alpha (L1)", args.xgb_reg_alpha, 0.0
                        )
                        step += 1
                        args.xgb_reg_lambda = self.prompt_float(
                            f"Step {step}: XGBoost reg lambda (L2)", args.xgb_reg_lambda, 0.0
                        )
                        step += 1
                    else:
                        args.xgb_subsample = 1.0
                        args.xgb_colsample = 1.0
                        args.xgb_reg_alpha = 0.0
                        args.xgb_reg_lambda = 1.0
                if args.model == "lgb":
                    args.lgb_estimators = self.prompt_int(
                        f"Step {step}: LightGBM estimators", args.lgb_estimators, 1
                    )
                    step += 1
                    args.lgb_max_depth = self.prompt_int(
                        f"Step {step}: LightGBM max depth (0 = unlimited)", args.lgb_max_depth, 0
                    )
                    step += 1
                    args.lgb_lr = self.prompt_float(
                        f"Step {step}: LightGBM learning rate", args.lgb_lr, 0.0
                    )
                    step += 1
                    if args.enable_regularization:
                        args.lgb_subsample = self.prompt_float(
                            f"Step {step}: LightGBM subsample", args.lgb_subsample, 0.0
                        )
                        step += 1
                        args.lgb_colsample = self.prompt_float(
                            f"Step {step}: LightGBM colsample_bytree", args.lgb_colsample, 0.0
                        )
                        step += 1
                        args.lgb_reg_alpha = self.prompt_float(
                            f"Step {step}: LightGBM reg alpha (L1)", args.lgb_reg_alpha, 0.0
                        )
                        step += 1
                        args.lgb_reg_lambda = self.prompt_float(
                            f"Step {step}: LightGBM reg lambda (L2)", args.lgb_reg_lambda, 0.0
                        )
                        step += 1
                    else:
                        args.lgb_subsample = 1.0
                        args.lgb_colsample = 1.0
                        args.lgb_reg_alpha = 0.0
                        args.lgb_reg_lambda = 0.0
        else:
            while True:
                encoder_step = step
                print("Step 2.2: Choose encoder")
                encoder_options = encoder_options_for_tokenizer(args.tokenizer)
                encoder_options = ["TF-IDF" if enc == "tfidf" else enc for enc in encoder_options]

                if args.encoder == "tfidf":
                    default_encoder = "TF-IDF"
                else:
                    default_encoder = args.encoder
                if default_encoder not in encoder_options:
                    default_encoder = encoder_options[0]
                chosen_encoder = self.prompt_choice(
                    "Encoder", encoder_options, default_encoder, allow_back=True
                )
                if chosen_encoder == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                if chosen_encoder == "TF-IDF":
                    args.encoder = "tfidf"
                else:
                    args.encoder = chosen_encoder
                step = encoder_step + 1

                step, transform_back = self._prompt_feature_transformation(
                    args, step, allow_back=True
                )
                if transform_back == "__BACK__":
                    print("Returning to previous step")
                    step = encoder_step
                    continue

                seq_encoder_compatible = [
                    "index",
                    "onehot",
                    "blosum62",
                    "pssm",
                    "pssm_blosum62_hybrid",
                    "psfm",
                    "psfm_blosum62_hybrid",
                    "word2vec",
                    "glove",
                    "fasttext",
                ]
                if normalize_task_type(args.task_type) == "pep_hlaii":
                    step, strategy_back = self._prompt_pair_strategy(
                        args,
                        step,
                        allow_single=args.encoder in seq_encoder_compatible,
                        allow_back=True,
                    )
                    if strategy_back == "__BACK__":
                        print("Returning to previous step")
                        step = encoder_step
                        continue
                    if args.pair_strategy == "single_stream":
                        step, route_back = self._prompt_sequence_route(args, step, allow_back=True)
                        if route_back == "__BACK__":
                            print("Returning to previous step")
                            step = encoder_step
                            continue
                        self._set_pair_route(args, pair_strategy="single_stream")
                    else:
                        step, route_back = self._prompt_two_stream_route(
                            args,
                            step,
                            allow_token=args.encoder in seq_encoder_compatible,
                            allow_back=True,
                        )
                        if route_back == "__BACK__":
                            print("Returning to previous step")
                            step = encoder_step
                            continue
                else:
                    family = encoder_family(args.encoder)
                    use_flattened_route = False
                    flattenable_sequence_encoders = {
                        "onehot",
                        "blosum62",
                        "pssm",
                        "pssm_blosum62_hybrid",
                        "psfm",
                        "psfm_blosum62_hybrid",
                        "word2vec",
                        "glove",
                        "fasttext",
                    }
                    if family == "sequence" and args.encoder in flattenable_sequence_encoders:
                        print(f"Step {step}: Choose predictor route")
                        route_options = [
                            "modeling architecture + prediction head",
                            "flattened features + classifier",
                        ]
                        chosen_route = self.prompt_choice(
                            "Predictor route", route_options, route_options[0], allow_back=True
                        )
                        if chosen_route == "__BACK__":
                            print("Returning to previous step")
                            step = encoder_step
                            continue
                        use_flattened_route = chosen_route == route_options[1]
                        step += 1
                    if family == "sequence" and not use_flattened_route:
                        step, route_back = self._prompt_sequence_route(args, step, allow_back=True)
                        if route_back == "__BACK__":
                            print("Returning to previous step")
                            step = encoder_step
                            continue
                    else:
                        classifier_options = (
                            CLASSICAL_CLASSIFIERS if use_flattened_route else CLASSIFIERS
                        )
                        step, route_back = self._prompt_classifier(
                            args, step, options=classifier_options, allow_back=True
                        )
                        if route_back == "__BACK__":
                            print("Returning to previous step")
                            step = encoder_step
                            continue
                if self._needs_ml_settings(args):
                    print(f"Step {step}: Choose ML feature scaling")
                    default_ml_scaler = args.ml_scaler if hasattr(args, "ml_scaler") else "standard"
                    ml_scaler_options = ["none", "standard", "minmax"]
                    if default_ml_scaler not in ml_scaler_options:
                        default_ml_scaler = "standard"
                    chosen_ml_scaler = self.prompt_choice(
                        "ML scaler", ml_scaler_options, default_ml_scaler, allow_back=True
                    )
                    if chosen_ml_scaler == "__BACK__":
                        print("Returning to previous step")
                        step = encoder_step
                        continue
                    args.ml_scaler = chosen_ml_scaler
                    step += 1
                    print(f"Step {step}: Choose ML dimensionality reduction")
                    default_ml_reduction = (
                        args.ml_reduction if hasattr(args, "ml_reduction") else "none"
                    )
                    ml_reduction_options = ["none", "pca", "kpca", "tsvd"]
                    if default_ml_reduction not in ml_reduction_options:
                        default_ml_reduction = "none"
                    chosen_ml_reduction = self.prompt_choice(
                        "ML reduction", ml_reduction_options, default_ml_reduction, allow_back=True
                    )
                    if chosen_ml_reduction == "__BACK__":
                        print("Returning to previous step")
                        step = encoder_step
                        continue
                    args.ml_reduction = chosen_ml_reduction
                    step += 1
                    if args.ml_reduction != "none":
                        default_dim = (
                            args.ml_reduction_dim if hasattr(args, "ml_reduction_dim") else 128
                        )
                        ml_dim = self.prompt_int(
                            f"Step {step}: ML reduction dimension", default_dim, 1, allow_back=True
                        )
                        if ml_dim == "__BACK__":
                            print("Returning to previous step")
                            step = encoder_step
                            continue
                        args.ml_reduction_dim = ml_dim
                        step += 1
                    if args.ml_reduction == "kpca":
                        default_kernel = args.kpca_kernel if hasattr(args, "kpca_kernel") else "rbf"
                        kpca_kernel = self.prompt_choice(
                            "KPCA kernel",
                            ["rbf", "linear", "poly", "cosine"],
                            default_kernel,
                            allow_back=True,
                        )
                        if kpca_kernel == "__BACK__":
                            print("Returning to previous step")
                            step = encoder_step
                            continue
                        args.kpca_kernel = kpca_kernel
                        step += 1
                        if kpca_kernel in {"rbf", "poly"}:
                            default_gamma = (
                                args.kpca_gamma
                                if getattr(args, "kpca_gamma", None) is not None
                                else 0.1
                            )
                            kpca_gamma = self.prompt_float(
                                f"Step {step}: KPCA gamma", default_gamma, 0.0, allow_back=True
                            )
                            if kpca_gamma == "__BACK__":
                                print("Returning to previous step")
                                step = encoder_step
                                continue
                            args.kpca_gamma = kpca_gamma
                            step += 1
                        else:
                            args.kpca_gamma = None
                else:
                    if hasattr(args, "ml_scaler"):
                        args.ml_scaler = "none"
                    if hasattr(args, "ml_reduction"):
                        args.ml_reduction = "none"
                break

            if args.tokenizer == "unigram":
                default_unigram_path = (
                    args.unigram_path
                    if hasattr(args, "unigram_path") and args.unigram_path
                    else f"{task_root}/unigram_tokenizers/unigram_tokenizer.json"
                )
                candidate = self.prompt_text(
                    f"Step {step}: Managed Unigram tokenizer path",
                    default_unigram_path,
                    allow_back=True,
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.unigram_path = candidate
                args.unigram_vocab_size = int(getattr(args, "unigram_vocab_size", 1000))
                args.unigram_sampling = bool(getattr(args, "unigram_sampling", False))
                args.unigram_nbest_size = int(getattr(args, "unigram_nbest_size", 64))
                args.unigram_alpha = float(getattr(args, "unigram_alpha", 0.1))
                args.unigram_max_raw_length = int(getattr(args, "unigram_max_raw_length", 50000))
                step += 1

            if args.tokenizer == "wordpiece":
                default_wordpiece_path = (
                    args.wordpiece_path
                    if hasattr(args, "wordpiece_path") and args.wordpiece_path
                    else f"{task_root}/wordpiece_tokenizers/wordpiece_tokenizer.json"
                )
                while True:
                    candidate = self.prompt_text(
                        f"Step {step}: WordPiece tokenizer path",
                        default_wordpiece_path,
                        allow_back=True,
                    )
                    if candidate == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    if os.path.isabs(candidate):
                        abs_candidate = candidate
                    else:
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        abs_candidate = os.path.join(project_root, candidate)
                    if os.path.exists(abs_candidate):
                        args.wordpiece_path = candidate
                        break
                    print(f"WordPiece tokenizer not found: {abs_candidate}")
                step += 1

            if (
                args.tokenizer == "kmer"
                and str(getattr(args, "encoder", "")).strip().lower() in COVARIANCE_ENCODERS
            ):
                args.k = 1
            elif args.tokenizer == "kmer":
                k_val = self.prompt_int(
                    f"Step {step}: k-mer size",
                    args.k,
                    1,
                    allow_back=True,
                    max_value=KMER_MAX_K,
                )
                if k_val == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.k = k_val
                step += 1

            if args.tokenizer == "dr":
                default_dr = args.dr_max_distance if hasattr(args, "dr_max_distance") else 5
                dr_max_distance = self.prompt_int(
                    f"Step {step}: DR max distance",
                    default_dr,
                    0,
                    allow_back=True,
                    max_value=DR_MAX_DISTANCE,
                )
                if dr_max_distance == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.dr_max_distance = dr_max_distance
                step += 1
                raw_length = self.prompt_int(
                    f"Step {step}: DR maximum raw sequence length",
                    getattr(
                        args,
                        "dr_max_raw_length",
                        DR_DEFAULT_MAX_RAW_LENGTH,
                    ),
                    1,
                    allow_back=True,
                    max_value=DR_ABSOLUTE_MAX_RAW_LENGTH,
                )
                if raw_length == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.dr_max_raw_length = raw_length
                step += 1

            if args.tokenizer == "top-n-gram":
                default_top_n = args.top_ngram_n if hasattr(args, "top_ngram_n") else 2
                maximum_top_n = 3
                while True:
                    top_n = self.prompt_int(
                        f"Step {step}: Top-N-Gram top_n", default_top_n, 1, allow_back=True
                    )
                    if top_n == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    if top_n < 1 or top_n > maximum_top_n:
                        print(f"top_ngram_n must be in [1,{maximum_top_n}], please reselect.")
                        continue
                    args.top_ngram_n = top_n
                    break

            if args.tokenizer in {
                "kmer",
                "dr",
                "top-n-gram",
                "dt",
            } and str(getattr(args, "encoder", "")).strip().lower() in {"bow", "tfidf", "tf-idf"}:
                raw_length = self.prompt_int(
                    f"Step {step}: BoW/TF-IDF maximum raw sequence length",
                    getattr(
                        args,
                        "frequency_max_raw_length",
                        FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                    ),
                    1,
                    allow_back=True,
                    max_value=FREQUENCY_MAX_RAW_LENGTH,
                )
                if raw_length == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.frequency_max_raw_length = raw_length
                step += 1
                step += 1

            if args.tokenizer == "dt":
                default_dt = args.dt_max_distance if hasattr(args, "dt_max_distance") else 5
                dt_max_distance = self.prompt_int(
                    f"Step {step}: DT max distance",
                    default_dt,
                    0,
                    allow_back=True,
                    max_value=DT_MAX_DISTANCE,
                )
                if dt_max_distance == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.dt_max_distance = dt_max_distance
                step += 1

            if args.encoder == "ac":
                default_ac_lag = args.ac_lag if hasattr(args, "ac_lag") else 10
                ac_lag = self.prompt_int(f"Step {step}: AC lag", default_ac_lag, 1, allow_back=True)
                if ac_lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.ac_lag = ac_lag
                step += 1

            if args.encoder == "cc":
                default_cc_lag = args.cc_lag if hasattr(args, "cc_lag") else 10
                cc_lag = self.prompt_int(f"Step {step}: CC lag", default_cc_lag, 1, allow_back=True)
                if cc_lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.cc_lag = cc_lag
                step += 1

            if args.encoder == "acc":
                default_acc_lag = args.acc_lag if hasattr(args, "acc_lag") else 10
                acc_lag = self.prompt_int(
                    f"Step {step}: ACC lag", default_acc_lag, 1, allow_back=True
                )
                if acc_lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.acc_lag = acc_lag
                step += 1

            if args.encoder in COVARIANCE_ENCODERS:
                raw_length = self.prompt_int(
                    f"Step {step}: AC/CC/ACC maximum raw sequence length",
                    getattr(
                        args,
                        "covariance_max_raw_length",
                        COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
                    ),
                    1,
                    allow_back=True,
                    max_value=COVARIANCE_MAX_RAW_LENGTH,
                )
                if raw_length == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.covariance_max_raw_length = raw_length
                step += 1

            if args.encoder == "pdt":
                default_pdt_lag = args.pdt_lag if hasattr(args, "pdt_lag") else PDT_DEFAULT_LAG
                pdt_lag = self.prompt_int(
                    f"Step {step}: PDT lag", default_pdt_lag, 1, allow_back=True
                )
                if pdt_lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.pdt_lag = pdt_lag
                step += 1
                pdt_max_raw_length = self.prompt_int(
                    f"Step {step}: PDT maximum raw sequence length",
                    getattr(
                        args,
                        "pdt_max_raw_length",
                        PDT_DEFAULT_MAX_RAW_LENGTH,
                    ),
                    1,
                    allow_back=True,
                    max_value=PDT_MAX_RAW_LENGTH,
                )
                if pdt_max_raw_length == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.pdt_max_raw_length = pdt_max_raw_length
                step += 1

            if args.encoder in {"pssm", "pssm_blosum62_hybrid"}:
                default_cache = (
                    "pssm_hybrid_cache" if args.encoder == "pssm_blosum62_hybrid" else "pssm_cache"
                )
                default_pssm_index = (
                    args.pssm_index_path
                    if hasattr(args, "pssm_index_path") and args.pssm_index_path
                    else f"{task_root}/{default_cache}/all/index.json"
                )
                candidate = self.prompt_text(
                    f"Step {step}: PSSM index path", default_pssm_index, allow_back=True
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.pssm_index_path = candidate or default_pssm_index
                if os.path.isabs(args.pssm_index_path):
                    abs_candidate = args.pssm_index_path
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    abs_candidate = os.path.join(project_root, args.pssm_index_path)
                if not os.path.exists(abs_candidate):
                    print(f"PSSM index not found: {abs_candidate}")
                    print(
                        f"Build cache first, e.g.: python tools/build_pssm_cache.py --data_dir {task_root}/data --split train --out_dir {task_root}/pssm_cache/all"
                    )
                args.profile_on_missing = (
                    "blosum62_hybrid" if args.encoder == "pssm_blosum62_hybrid" else "error"
                )
                step += 1

            if args.encoder == "ac_pssm":
                default_pssm_index = (
                    args.pssm_index_path
                    if hasattr(args, "pssm_index_path") and args.pssm_index_path
                    else f"{task_root}/pssm_cache/all/index.json"
                )
                candidate = self.prompt_text(
                    f"Step {step}: PSSM index path", default_pssm_index, allow_back=True
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.pssm_index_path = candidate or default_pssm_index
                if os.path.isabs(args.pssm_index_path):
                    abs_candidate = args.pssm_index_path
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    abs_candidate = os.path.join(project_root, args.pssm_index_path)
                if not os.path.exists(abs_candidate):
                    print(f"PSSM index not found: {abs_candidate}")
                    print(
                        f"Build cache first, e.g.: python tools/build_pssm_cache.py --data_dir {task_root}/data --split train --out_dir {task_root}/pssm_cache/all"
                    )
                step += 1
                default_ac_pssm_lag = args.ac_pssm_lag if hasattr(args, "ac_pssm_lag") else 10
                ac_pssm_lag = self.prompt_int(
                    f"Step {step}: AC-PSSM lag", default_ac_pssm_lag, 1, allow_back=True
                )
                if ac_pssm_lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.ac_pssm_lag = ac_pssm_lag
                step += 1

            if args.encoder == "cc_pssm":
                default_pssm_index = (
                    args.pssm_index_path
                    if hasattr(args, "pssm_index_path") and args.pssm_index_path
                    else f"{task_root}/pssm_cache/all/index.json"
                )
                candidate = self.prompt_text(
                    f"Step {step}: PSSM index path", default_pssm_index, allow_back=True
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.pssm_index_path = candidate or default_pssm_index
                if os.path.isabs(args.pssm_index_path):
                    abs_candidate = args.pssm_index_path
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    abs_candidate = os.path.join(project_root, args.pssm_index_path)
                if not os.path.exists(abs_candidate):
                    print(f"PSSM index not found: {abs_candidate}")
                    print(
                        f"Build cache first, e.g.: python tools/build_pssm_cache.py --data_dir {task_root}/data --split train --out_dir {task_root}/pssm_cache/all"
                    )
                step += 1
                default_cc_pssm_lag = args.cc_pssm_lag if hasattr(args, "cc_pssm_lag") else 10
                cc_pssm_lag = self.prompt_int(
                    f"Step {step}: CC-PSSM lag", default_cc_pssm_lag, 1, allow_back=True
                )
                if cc_pssm_lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.cc_pssm_lag = cc_pssm_lag
                step += 1

            if args.encoder == "acc_pssm":
                default_pssm_index = (
                    args.pssm_index_path
                    if hasattr(args, "pssm_index_path") and args.pssm_index_path
                    else f"{task_root}/pssm_cache/all/index.json"
                )
                candidate = self.prompt_text(
                    f"Step {step}: PSSM index path", default_pssm_index, allow_back=True
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.pssm_index_path = candidate or default_pssm_index
                if os.path.isabs(args.pssm_index_path):
                    abs_candidate = args.pssm_index_path
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    abs_candidate = os.path.join(project_root, args.pssm_index_path)
                if not os.path.exists(abs_candidate):
                    print(f"PSSM index not found: {abs_candidate}")
                    print(
                        f"Build cache first, e.g.: python tools/build_pssm_cache.py --data_dir {task_root}/data --split train --out_dir {task_root}/pssm_cache/all"
                    )
                step += 1
                default_acc_pssm_lag = args.acc_pssm_lag if hasattr(args, "acc_pssm_lag") else 10
                acc_pssm_lag = self.prompt_int(
                    f"Step {step}: ACC-PSSM lag", default_acc_pssm_lag, 1, allow_back=True
                )
                if acc_pssm_lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.acc_pssm_lag = acc_pssm_lag
                step += 1

            if args.encoder in HYBRID_PSSM_COVARIANCE_ENCODERS:
                index_key = f"{args.encoder}_index_path"
                default_index = (
                    getattr(args, index_key, "")
                    or f"{task_root}/profile_cache/{args.encoder}/index.json"
                )
                candidate = self.prompt_text(
                    f"Step {step}: {args.encoder} index path",
                    default_index,
                    allow_back=True,
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                setattr(args, index_key, candidate or default_index)
                args.profile_on_missing = "blosum62_hybrid"
                step += 1
                lag_key = pssm_covariance_lag_key(args.encoder)
                lag = self.prompt_int(
                    f"Step {step}: {args.encoder} lag",
                    getattr(args, lag_key, 10),
                    1,
                    allow_back=True,
                )
                if lag == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                setattr(args, lag_key, lag)
                step += 1

            if args.encoder in {"psfm", "psfm_blosum62_hybrid"}:
                default_psfm_index = (
                    args.psfm_index_path
                    if hasattr(args, "psfm_index_path") and args.psfm_index_path
                    else f"{task_root}/{'psfm_hybrid_cache' if args.encoder == 'psfm_blosum62_hybrid' else 'psfm_cache'}/all/index.json"
                )
                candidate = self.prompt_text(
                    f"Step {step}: PSFM index path", default_psfm_index, allow_back=True
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.psfm_index_path = candidate or default_psfm_index
                args.profile_on_missing = (
                    "blosum62_hybrid" if args.encoder == "psfm_blosum62_hybrid" else "error"
                )
                if os.path.isabs(args.psfm_index_path):
                    abs_candidate = args.psfm_index_path
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    abs_candidate = os.path.join(project_root, args.psfm_index_path)
                if not os.path.exists(abs_candidate):
                    print(f"PSFM index not found: {abs_candidate}")
                step += 1

            if args.encoder == "word2vec":
                default_word2vec = (
                    args.word2vec_path
                    if hasattr(args, "word2vec_path") and args.word2vec_path
                    else f"{task_root}/word2vec/word2vec.vec"
                )
                candidate = self.prompt_text(
                    f"Step {step}: Word2Vec embedding path", default_word2vec, allow_back=True
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.word2vec_path = candidate or default_word2vec
                if os.path.isabs(args.word2vec_path):
                    abs_candidate = args.word2vec_path
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    abs_candidate = os.path.join(project_root, args.word2vec_path)
                if not os.path.exists(abs_candidate):
                    print(f"Word2Vec embedding not found: {abs_candidate}")
                    print("Will auto-build Word2Vec embedding cache before training.")
                step += 1

            if args.encoder == "glove":
                default_glove = (
                    args.glove_path if hasattr(args, "glove_path") and args.glove_path else ""
                )
                candidate = self.prompt_text(
                    f"Step {step}: GloVe embedding path (leave blank to auto-train)",
                    default_glove,
                    allow_back=True,
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.glove_path = candidate
                if args.glove_path:
                    if os.path.isabs(args.glove_path):
                        abs_candidate = args.glove_path
                    else:
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        abs_candidate = os.path.join(project_root, args.glove_path)
                    if not os.path.exists(abs_candidate):
                        print(f"GloVe embedding not found: {abs_candidate}")
                else:
                    print("Will auto-build the GloVe embedding cache from the training split.")
                step += 1

            if args.encoder == "fasttext":
                default_fasttext = (
                    args.fasttext_path
                    if hasattr(args, "fasttext_path") and args.fasttext_path
                    else f"{task_root}/fasttext/fasttext.bin"
                )
                candidate = self.prompt_text(
                    f"Step {step}: FastText embedding path", default_fasttext, allow_back=True
                )
                if candidate == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.fasttext_path = candidate or default_fasttext
                if os.path.isabs(args.fasttext_path):
                    abs_candidate = args.fasttext_path
                else:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    abs_candidate = os.path.join(project_root, args.fasttext_path)
                if not os.path.exists(abs_candidate):
                    print(f"FastText embedding not found: {abs_candidate}")
                    print("Will auto-build FastText embedding cache before training.")
                step += 1

            args.max_length = self.prompt_int(f"Step {step}: max length", args.max_length, 1)
            step += 1

            if self._is_trainable_dl_route(args):
                batch_size = self.prompt_int(
                    f"Step {step}: batch_size", args.batch_size, 1, allow_back=True
                )
                if batch_size == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.batch_size = batch_size
                step += 1
                lr = self.prompt_float(f"Step {step}: learning rate", args.lr, 0.0, allow_back=True)
                if lr == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.lr = lr
                step += 1
                epochs = self.prompt_int(f"Step {step}: epochs", args.epochs, 1, allow_back=True)
                if epochs == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.epochs = epochs
                step += 1
                if args.encoder != "index":
                    dl_scaler = self.prompt_choice(
                        f"Step {step}: DL feature scaling",
                        ["none", "standard"],
                        args.dl_scaler,
                        allow_back=True,
                    )
                    if dl_scaler == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.dl_scaler = dl_scaler
                    step += 1
                else:
                    args.dl_scaler = "none"

            step, semantic_back = self._prompt_semantic_auxiliary(args, step, allow_back=True)
            if semantic_back == "__BACK__":
                print("Returning to previous step")
                return self.run(args)

            if self._regularization_relevant(args):
                default_enable_reg = "yes" if getattr(args, "enable_regularization", True) else "no"
                enable_reg = self.prompt_choice(
                    f"Step {step}: Enable regularization",
                    ["yes", "no"],
                    default_enable_reg,
                    allow_back=True,
                )
                if enable_reg == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.enable_regularization = enable_reg == "yes"
                step += 1
            else:
                args.enable_regularization = False

            if self._is_trainable_dl_route(args):
                if args.enable_regularization:
                    dropout = self.prompt_float(
                        f"Step {step}: dropout", args.dropout, 0.0, allow_back=True
                    )
                    if dropout == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.dropout = dropout
                    step += 1
                    weight_decay = self.prompt_float(
                        f"Step {step}: weight decay", args.weight_decay, 0.0, allow_back=True
                    )
                    if weight_decay == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.weight_decay = weight_decay
                    step += 1
                    grad_clip_norm = self.prompt_float(
                        f"Step {step}: grad clip norm (0=off)",
                        args.grad_clip_norm,
                        0.0,
                        allow_back=True,
                    )
                    if grad_clip_norm == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.grad_clip_norm = grad_clip_norm
                    step += 1
                    early_stop_patience = self.prompt_int(
                        f"Step {step}: early stop patience (0=off)",
                        args.early_stop_patience,
                        0,
                        allow_back=True,
                    )
                    if early_stop_patience == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.early_stop_patience = early_stop_patience
                    step += 1
                    early_stop_min_delta = self.prompt_float(
                        f"Step {step}: early stop min delta",
                        args.early_stop_min_delta,
                        0.0,
                        allow_back=True,
                    )
                    if early_stop_min_delta == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.early_stop_min_delta = early_stop_min_delta
                    step += 1
                else:
                    args.dropout = 0.0
                    args.weight_decay = 0.0
                    args.grad_clip_norm = 0.0
                    args.early_stop_patience = 0
                    args.early_stop_min_delta = 0.0

            step, head_back = self._prompt_prediction_head(args, step, allow_back=True)
            if head_back == "__BACK__":
                print("Returning to previous step")
                return self.run(args)

            downstream_classifier = effective_classifier(args)
            if downstream_classifier == "svm":
                svm_calibration = self.prompt_choice(
                    f"Step {step}: SVM calibration",
                    ["none", "sigmoid"],
                    getattr(args, "svm_calibration", "none"),
                    allow_back=True,
                )
                if svm_calibration == "__BACK__":
                    print("Returning to previous step")
                    return self.run(args)
                args.svm_calibration = svm_calibration
                args.svm_calibration_cv_folds = 3
                step += 1
                if args.enable_regularization:
                    svm_class_weight = self.prompt_choice(
                        f"Step {step}: SVM class weight",
                        ["balanced", "none"],
                        args.svm_class_weight,
                        allow_back=True,
                    )
                    if svm_class_weight == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.svm_class_weight = svm_class_weight
                    step += 1
                else:
                    args.svm_class_weight = "none"

            if downstream_classifier == "logreg":
                if args.enable_regularization:
                    logreg_class_weight = self.prompt_choice(
                        f"Step {step}: LogReg class weight",
                        ["balanced", "none"],
                        args.logreg_class_weight,
                        allow_back=True,
                    )
                    if logreg_class_weight == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.logreg_class_weight = logreg_class_weight
                    step += 1
                else:
                    args.logreg_class_weight = "none"
                args.logreg_c = self.prompt_float(f"Step {step}: LogReg C", args.logreg_c, 1e-12)
                step += 1
                args.logreg_max_iter = self.prompt_int(
                    f"Step {step}: LogReg max_iter", args.logreg_max_iter, 1
                )
                step += 1

            if downstream_classifier == "rf":
                args.rf_estimators = self.prompt_int(
                    f"Step {step}: RF estimators", args.rf_estimators, 1
                )
                step += 1
                args.rf_max_depth = self.prompt_int(
                    f"Step {step}: RF max depth (0 = unlimited)", args.rf_max_depth, 0
                )
                step += 1
                args.rf_min_samples_leaf = self.prompt_int(
                    f"Step {step}: RF min samples per leaf", args.rf_min_samples_leaf, 1
                )
                step += 1

            if downstream_classifier == "extratrees":
                args.et_estimators = self.prompt_int(
                    f"Step {step}: ExtraTrees estimators", args.et_estimators, 1
                )
                step += 1
                args.et_max_depth = self.prompt_int(
                    f"Step {step}: ExtraTrees max depth (0 = unlimited)", args.et_max_depth, 0
                )
                step += 1
                args.et_min_samples_leaf = self.prompt_int(
                    f"Step {step}: ExtraTrees min samples per leaf", args.et_min_samples_leaf, 1
                )
                step += 1

            if downstream_classifier == "histgb":
                args.histgb_max_iter = self.prompt_int(
                    f"Step {step}: HistGB max_iter", args.histgb_max_iter, 1
                )
                step += 1
                while True:
                    args.histgb_max_leaf_nodes = self.prompt_int(
                        f"Step {step}: HistGB max leaf nodes (0 = unlimited)",
                        args.histgb_max_leaf_nodes,
                        0,
                    )
                    if args.histgb_max_leaf_nodes == 0 or args.histgb_max_leaf_nodes >= 2:
                        break
                    print("HistGB max leaf nodes must be 0 (unlimited) or at least 2.")
                step += 1
                args.histgb_lr = self.prompt_float(
                    f"Step {step}: HistGB learning rate", args.histgb_lr, 1e-12
                )
                step += 1
                if args.enable_regularization:
                    args.histgb_l2_regularization = self.prompt_float(
                        f"Step {step}: HistGB L2 regularization", args.histgb_l2_regularization, 0.0
                    )
                    step += 1
                else:
                    args.histgb_l2_regularization = 0.0

            if downstream_classifier == "xgb":
                args.xgb_estimators = self.prompt_int(
                    f"Step {step}: XGBoost estimators", args.xgb_estimators, 1
                )
                step += 1
                args.xgb_max_depth = self.prompt_int(
                    f"Step {step}: XGBoost max depth (0 = unlimited)", args.xgb_max_depth, 0
                )
                step += 1
                args.xgb_lr = self.prompt_float(
                    f"Step {step}: XGBoost learning rate", args.xgb_lr, 0.0
                )
                step += 1
                if args.enable_regularization:
                    xgb_subsample = self.prompt_float(
                        f"Step {step}: XGBoost subsample", args.xgb_subsample, 0.0, allow_back=True
                    )
                    if xgb_subsample == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.xgb_subsample = xgb_subsample
                    step += 1
                    xgb_colsample = self.prompt_float(
                        f"Step {step}: XGBoost colsample_bytree",
                        args.xgb_colsample,
                        0.0,
                        allow_back=True,
                    )
                    if xgb_colsample == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.xgb_colsample = xgb_colsample
                    step += 1
                    xgb_reg_alpha = self.prompt_float(
                        f"Step {step}: XGBoost reg alpha (L1)",
                        args.xgb_reg_alpha,
                        0.0,
                        allow_back=True,
                    )
                    if xgb_reg_alpha == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.xgb_reg_alpha = xgb_reg_alpha
                    step += 1
                    xgb_reg_lambda = self.prompt_float(
                        f"Step {step}: XGBoost reg lambda (L2)",
                        args.xgb_reg_lambda,
                        0.0,
                        allow_back=True,
                    )
                    if xgb_reg_lambda == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.xgb_reg_lambda = xgb_reg_lambda
                    step += 1
                else:
                    args.xgb_subsample = 1.0
                    args.xgb_colsample = 1.0
                    args.xgb_reg_alpha = 0.0
                    args.xgb_reg_lambda = 1.0

            if downstream_classifier == "lgb":
                args.lgb_estimators = self.prompt_int(
                    f"Step {step}: LightGBM estimators", args.lgb_estimators, 1
                )
                step += 1
                args.lgb_max_depth = self.prompt_int(
                    f"Step {step}: LightGBM max depth (0 = unlimited)", args.lgb_max_depth, 0
                )
                step += 1
                args.lgb_lr = self.prompt_float(
                    f"Step {step}: LightGBM learning rate", args.lgb_lr, 0.0
                )
                step += 1
                if args.enable_regularization:
                    lgb_subsample = self.prompt_float(
                        f"Step {step}: LightGBM subsample", args.lgb_subsample, 0.0, allow_back=True
                    )
                    if lgb_subsample == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.lgb_subsample = lgb_subsample
                    step += 1
                    lgb_colsample = self.prompt_float(
                        f"Step {step}: LightGBM colsample_bytree",
                        args.lgb_colsample,
                        0.0,
                        allow_back=True,
                    )
                    if lgb_colsample == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.lgb_colsample = lgb_colsample
                    step += 1
                    lgb_reg_alpha = self.prompt_float(
                        f"Step {step}: LightGBM reg alpha (L1)",
                        args.lgb_reg_alpha,
                        0.0,
                        allow_back=True,
                    )
                    if lgb_reg_alpha == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.lgb_reg_alpha = lgb_reg_alpha
                    step += 1
                    lgb_reg_lambda = self.prompt_float(
                        f"Step {step}: LightGBM reg lambda (L2)",
                        args.lgb_reg_lambda,
                        0.0,
                        allow_back=True,
                    )
                    if lgb_reg_lambda == "__BACK__":
                        print("Returning to previous step")
                        return self.run(args)
                    args.lgb_reg_lambda = lgb_reg_lambda
                    step += 1
                else:
                    args.lgb_subsample = 1.0
                    args.lgb_colsample = 1.0
                    args.lgb_reg_alpha = 0.0
                    args.lgb_reg_lambda = 0.0

        default_data_dir = self._default_data_dir(args, task_root)
        data_dir = self.prompt_text(
            f"Step {step}: Data directory", default_data_dir, allow_back=True
        )
        if data_dir == "__BACK__":
            print("Returning to previous step")
            return self.run(args)
        args.data_dir = data_dir
        print("Configuration complete")
        print(f"tokenizer={args.tokenizer}")
        if args.tokenizer == "unigram":
            print(f"unigram_path={args.unigram_path}")
        if args.tokenizer == "wordpiece":
            print(f"wordpiece_path={args.wordpiece_path}")
        if args.tokenizer == "protbert":
            print(f"protbert_feature_dir={args.protbert_feature_dir}")
            print(f"protbert_model={args.protbert_model}")
        if args.tokenizer == "prott5":
            print(f"prott5_feature_dir={args.prott5_feature_dir}")
            print(f"prott5_model={args.prott5_model}")
        if args.tokenizer == "esmc":
            print(f"esmc_feature_dir={args.esmc_feature_dir}")
            print(f"esmc_model={args.esmc_model}")
        if args.tokenizer == "esm3":
            print(f"esm3_feature_dir={args.esm3_feature_dir}")
            print(f"esm3_model={args.esm3_model}")

        if args.encoder == "word2vec":
            print(f"word2vec_path={args.word2vec_path}")
        if args.encoder == "glove":
            print(f"glove_path={args.glove_path}")
        if args.encoder == "fasttext":
            print(f"fasttext_path={args.fasttext_path}")
        if args.encoder in {"pssm", "pssm_blosum62_hybrid"}:
            print(f"pssm_index_path={args.pssm_index_path}")
        if args.encoder == "ac_pssm":
            print(f"pssm_index_path={args.pssm_index_path}")
        if args.encoder == "cc_pssm":
            print(f"pssm_index_path={args.pssm_index_path}")
        if args.encoder == "acc_pssm":
            print(f"pssm_index_path={args.pssm_index_path}")
        if args.encoder in HYBRID_PSSM_COVARIANCE_ENCODERS:
            index_key = f"{args.encoder}_index_path"
            print(f"{index_key}={getattr(args, index_key)}")
        if args.encoder in {"psfm", "psfm_blosum62_hybrid"}:
            print(f"psfm_index_path={args.psfm_index_path}")
        if args.tokenizer in {"top-n-gram", "dt"}:
            print(f"frequency_profile_index_path={args.frequency_profile_index_path}")
        if args.encoder == "tfidf":
            print("encoder=TF-IDF")
        else:
            print(f"encoder={args.encoder}")
        if args.encoder == "ac":
            print(f"ac_lag={args.ac_lag}")
        if args.encoder == "cc":
            print(f"cc_lag={args.cc_lag}")
        if args.encoder == "acc":
            print(f"acc_lag={args.acc_lag}")
        if args.encoder in COVARIANCE_ENCODERS:
            print(
                "covariance_max_raw_length="
                f"{getattr(args, 'covariance_max_raw_length', COVARIANCE_DEFAULT_MAX_RAW_LENGTH)}"
            )
        if args.encoder == "pdt":
            print(f"pdt_lag={args.pdt_lag}")
            print(
                "pdt_max_raw_length="
                f"{getattr(args, 'pdt_max_raw_length', PDT_DEFAULT_MAX_RAW_LENGTH)}"
            )
        if args.encoder == "ac_pssm":
            print(f"ac_pssm_lag={args.ac_pssm_lag}")
        if args.encoder == "cc_pssm":
            print(f"cc_pssm_lag={args.cc_pssm_lag}")
        if args.encoder == "acc_pssm":
            print(f"acc_pssm_lag={args.acc_pssm_lag}")
        if args.encoder in HYBRID_PSSM_COVARIANCE_ENCODERS:
            lag_key = pssm_covariance_lag_key(args.encoder)
            print(f"{lag_key}={getattr(args, lag_key)}")
        print(f"model={args.model}")
        if normalize_task_type(getattr(args, "task_type", "")) == "pep_hlaii":
            print(f"pair_strategy={public_pair_strategy(getattr(args, 'pair_strategy', ''))}")
            if getattr(args, "interaction_level", ""):
                print(f"interaction_level={args.interaction_level}")
            if getattr(args, "interaction_model", ""):
                print(f"interaction_model={args.interaction_model}")
            print(f"predictor_type={getattr(args, 'predictor_type', 'classifier')}")
            if getattr(args, "graph_predictor", ""):
                print(f"graph_predictor={args.graph_predictor}")
        if getattr(args, "sequence_model", ""):
            print(f"sequence_model={args.sequence_model}")
            print(f"training_mode={args.training_mode}")
            print(f"classifier={args.classifier}")
        if getattr(args, "head_hidden_dims", None) is not None:
            print(f"head_hidden_dims={json.dumps(args.head_hidden_dims)}")
            if args.head_hidden_dims:
                print(f"head_activation={args.head_activation}")
                print(f"head_dropout={args.head_dropout}")
        if args.reduction != "none":
            print(f"reduction={args.reduction}")
            if args.reduction == "lsa":
                print(f"lsa_dim={args.lsa_dim}")
                print(f"lsa_dense_memory_limit_gib={args.lsa_dense_memory_limit_gib}")
            if args.reduction == "lda":
                print(f"lda_topics={args.lda_topics}")
                print(f"lda_dense_memory_limit_gib={args.lda_dense_memory_limit_gib}")
            if args.reduction == "plsa":
                print(f"plsa_topics={args.plsa_topics}")
        print(f"k={args.k}")
        if args.tokenizer == "dr":
            print(f"dr_max_distance={args.dr_max_distance}")
            print(
                f"dr_max_raw_length={getattr(args, 'dr_max_raw_length', DR_DEFAULT_MAX_RAW_LENGTH)}"
            )
        if args.tokenizer == "top-n-gram":
            print(f"top_ngram_n={args.top_ngram_n}")
        if args.tokenizer == "dt":
            print("top_ngram_n=1 (fixed by DT v1)")
            print(f"dt_max_distance={args.dt_max_distance}")
        if args.tokenizer in {
            "kmer",
            "dr",
            "top-n-gram",
            "dt",
        } and str(getattr(args, "encoder", "")).strip().lower() in {"bow", "tfidf", "tf-idf"}:
            print(
                "frequency_max_raw_length="
                f"{getattr(args, 'frequency_max_raw_length', FREQUENCY_DEFAULT_MAX_RAW_LENGTH)}"
            )

        print(f"max_length={args.max_length}")
        if self._is_trainable_dl_route(args):
            print(f"batch_size={args.batch_size}")
            print(f"lr={args.lr}")
            print(f"epochs={args.epochs}")
            print(f"dl_scaler={args.dl_scaler}")
            print(f"enable_semantic_similarity={self._semantic_auxiliary_enabled(args)}")
            if self._semantic_auxiliary_enabled(args):
                print("semantic_similarity_type=auxiliary_loss")
            print(f"enable_regularization={args.enable_regularization}")
            if args.enable_regularization:
                print(f"dropout={args.dropout}")
                print(f"weight_decay={args.weight_decay}")
                print(f"grad_clip_norm={args.grad_clip_norm}")
                print(f"early_stop_patience={args.early_stop_patience}")
                print(f"early_stop_min_delta={args.early_stop_min_delta}")
        print(f"data_dir={args.data_dir}")
        downstream_classifier = effective_classifier(args)
        if downstream_classifier in CLASSICAL_CLASSIFIERS:
            print(f"ml_scaler={args.ml_scaler}")
            print(f"ml_reduction={args.ml_reduction}")
            if args.ml_reduction != "none":
                print(f"ml_reduction_dim={args.ml_reduction_dim}")
            if args.ml_reduction == "tsvd":
                print(
                    "tsvd_dense_memory_limit_gib="
                    f"{getattr(args, 'tsvd_dense_memory_limit_gib', 4.0)}"
                )
            if args.ml_reduction == "pca":
                print(
                    f"pca_dense_memory_limit_gib={getattr(args, 'pca_dense_memory_limit_gib', 4.0)}"
                )
            if args.ml_reduction == "kpca":
                print(f"kpca_kernel={args.kpca_kernel}")
                print(f"kpca_memory_limit_gib={getattr(args, 'kpca_memory_limit_gib', 4.0)}")
                if args.kpca_kernel in {"rbf", "poly"}:
                    print(f"kpca_gamma={args.kpca_gamma}")
        if downstream_classifier == "rf":
            print(f"rf_estimators={args.rf_estimators}")
            print(f"rf_max_depth={args.rf_max_depth}")
            print(f"rf_min_samples_leaf={args.rf_min_samples_leaf}")
            print(f"enable_regularization={args.enable_regularization}")
        if downstream_classifier == "svm":
            print(f"svm_calibration={getattr(args, 'svm_calibration', 'none')}")
            print(f"svm_calibration_cv_folds={getattr(args, 'svm_calibration_cv_folds', 3)}")
            print(f"enable_regularization={args.enable_regularization}")
            if args.enable_regularization:
                print(f"svm_class_weight={args.svm_class_weight}")
        if downstream_classifier == "extratrees":
            print(f"et_estimators={args.et_estimators}")
            print(f"et_max_depth={args.et_max_depth}")
            print(f"et_min_samples_leaf={args.et_min_samples_leaf}")
        if downstream_classifier == "histgb":
            print(f"histgb_max_iter={args.histgb_max_iter}")
            print(f"histgb_max_leaf_nodes={args.histgb_max_leaf_nodes}")
            print(f"histgb_lr={args.histgb_lr}")
            print(f"enable_regularization={args.enable_regularization}")
            if args.enable_regularization:
                print(f"histgb_l2_regularization={args.histgb_l2_regularization}")
        if downstream_classifier == "xgb":
            print(f"xgb_estimators={args.xgb_estimators}")
            print(f"xgb_max_depth={args.xgb_max_depth}")
            print(f"xgb_lr={args.xgb_lr}")
            print(f"enable_regularization={args.enable_regularization}")
            if args.enable_regularization:
                print(f"xgb_subsample={args.xgb_subsample}")
                print(f"xgb_colsample={args.xgb_colsample}")
                print(f"xgb_reg_alpha={args.xgb_reg_alpha}")
                print(f"xgb_reg_lambda={args.xgb_reg_lambda}")
        if downstream_classifier == "logreg":
            print(f"logreg_c={args.logreg_c}")
            print(f"logreg_max_iter={args.logreg_max_iter}")
            print(
                f"logreg_convergence_policy={getattr(args, 'logreg_convergence_policy', 'strict')}"
            )
            print(f"enable_regularization={args.enable_regularization}")
            if args.enable_regularization:
                print(f"logreg_class_weight={args.logreg_class_weight}")
        return args


from pepplatform.stream_compat import install_legacy_exports as _install_legacy_stream_exports
_install_legacy_stream_exports(globals())
