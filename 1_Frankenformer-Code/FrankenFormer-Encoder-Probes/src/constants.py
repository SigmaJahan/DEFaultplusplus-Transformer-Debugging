"""Global constants for ABNN Encoder Fault Injection Pipeline."""

DEFAULT_BATCH_SIZE = 16
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_MAX_LENGTH = 128
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOGGING_STEPS = 50
DEFAULT_SEED_LIST = [42, 123, 456, 789, 101112]
DEFAULT_SINGLE_SEED = 42

BASE_LAYER_DEPTH = 6
MAX_POSITION_EMBEDDINGS = 512

DEFAULT_INVARIANCE_PROBE_BATCHES = 4
DEFAULT_EDGE_PROBE_BATCHES = 2
DEFAULT_INVARIANCE_PAD_TOKENS = 2

MIN_ATTENTION_ENTROPY = 1e-10
MIN_VARIANCE_THRESHOLD = 1e-8
EPSILON_NUMERICAL_STABILITY = 1e-10

DEFAULT_HDF5_LOCK_TIMEOUT = 300.0
DEFAULT_ENABLE_HDF5_LOCKING = True

DEFAULT_QUERY_BATCH_SIZE = 1000

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_SEED_COUNT = 3
DEFAULT_PERMUTATION_SAMPLES = 10000

SMALL_EFFECT_SIZE = 0.2
MEDIUM_EFFECT_SIZE = 0.5
LARGE_EFFECT_SIZE = 0.8

DEFAULT_MASTER_CONFIG = "config/master_config.yaml"
DEFAULT_PIPELINE_CONFIG = "config/pipeline_configs_probes.json"
DEFAULT_MATRIX_CONFIG = "config/matrix_encoder.yaml"

DEFAULT_RESULTS_DIR = "results"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_SLURM_LOGS_DIR = "slurm-logs"

METRICS_FILENAME = "metrics.h5"
DATABASE_FILENAME = "dataset.db"
KILL_RESULTS_CSV = "kill_evaluation_results.csv"
KILL_SUMMARY_CSV = "kill_evaluation_summary.csv"

GPU_CACHE_CLEAR_FREQUENCY = 100
GARBAGE_COLLECTION_FREQUENCY = 50

MODEL_LOAD_TIMEOUT = 600
DATASET_LOAD_TIMEOUT = 300
TRAINING_STEP_TIMEOUT = 120

SUPPORTED_ARCHITECTURES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "google/electra-small-discriminator",
]

LAYER_EXTRACTION_PATHS = [
    ("bert", "encoder", "layer"),
    ("distilbert", "transformer", "layer"),
    ("roberta", "encoder", "layer"),
    ("electra", "encoder", "layer"),
    ("model", "encoder", "layer"),
]

FAULT_LAYER_GROUPS = {
    "group_1": [0, 2, 4],
    "group_2": [1, 3, 5],
    "group_3": [0, 3, 5],
    "group_4": [1, 2, 4],
    "group_5": [0, 2, 5],
}

FAULT_CATEGORIES = [
    "masking", "qkv", "score", "positional", "kernel",
    "variant", "ffn", "layernorm", "residual", "embedding", "output",
]

CONFIG_STATUS_PENDING = "pending"
CONFIG_STATUS_RUNNING = "running"
CONFIG_STATUS_COMPLETED = "completed"
CONFIG_STATUS_FAILED = "failed"
CONFIG_STATUS_SKIPPED = "skipped"

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 48
MIN_LEARNING_RATE = 1e-8
MAX_LEARNING_RATE = 1.0
MIN_EPOCHS = 1
MAX_EPOCHS = 10
MIN_LAYER_INDEX = 0
MAX_LAYER_INDEX = 100

DEFAULT_SLURM_MEMORY = "24G"
DEFAULT_SLURM_TIME = "12:00:00"
DEFAULT_SLURM_CPUS = 4
DEFAULT_SLURM_GPUS = 1

DEFAULT_PYTHON_VERSION = "3.10"
DEFAULT_CUDA_VERSION = "12.2"
DEFAULT_STDENV_VERSION = "2023"

DEFAULT_ENCODER_BATCH_SIZE = 16
DEFAULT_ENCODER_MAX_LENGTH = 128
DEFAULT_ENCODER_LEARNING_RATE = 2e-5

ENCODER_TASK_TYPES = [
    'cls_sst2',
    'cls_mnli',
    'cls_qqp',
    'cls_cola',
    'cls_mrpc',
    'cls_rte',
    'cls_stsb',
    'ner_conll2003',
    'mlm',
]

SUPPORTED_ENCODER_ARCHITECTURES = [
    "bert", "distilbert", "roberta", "electra", "modernbert"
]

ENCODER_LAYER_EXTRACTION_PATHS = [
    ("bert", "encoder", "layer"),
    ("distilbert", "transformer", "layer"),
    ("roberta", "encoder", "layer"),
    ("electra", "encoder", "layer"),
]

ENCODER_FAULT_CATEGORIES = FAULT_CATEGORIES + ["pooler"]

METRIC_ID_CLS_ACCURACY = 20
METRIC_ID_CLS_F1 = 21
METRIC_ID_CLS_PRECISION = 22
METRIC_ID_CLS_RECALL = 23
METRIC_ID_CLS_AUC = 24
METRIC_ID_MLM_ACCURACY = 25
METRIC_ID_MLM_PERPLEXITY = 26

DEFAULT_CLASSIFICATION_NUM_LABELS = 2

DEFAULT_ENCODER_MATRIX_CONFIG = "config/matrix_encoder.yaml"
DEFAULT_ENCODER_FAULT_CONFIG = "config/encoder_fault_configs.yaml"
