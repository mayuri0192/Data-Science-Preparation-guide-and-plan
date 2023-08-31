
config=dict(
    # RUN CONFIG:
    RUN_NAME='unofficial_overfit_cpu_run',
    RUN_DESCRIPTION="""
        This run is for testing can the model overfit a single example.
        It is useful when debugging.
        For better results change the scheduler in train.py.
        """,
    RUNS_FOLDER_PTH='../runs',
    # DATA CONFIG:
    DATASET_SIZE=2,
    TEST_PROPORTION=0.5,
    MAX_SEQ_LEN=100,
    VOCAB_SIZE=100,
    TOKENIZER_TYPE='wordlevel', # 'wordlevel' or 'bpe
    # TRAINING CONFIG:
    BATCH_SIZE=1, 
    GRAD_ACCUMULATION_STEPS=1,
    WORKER_COUNT=10,
    EPOCHS=1000,
    # OPTIMIZER CONFIG:
    BETAS=(0.9, 0.98),
    EPS=1e-9,
    # SCHEDULER CONFIG:
    N_WARMUP_STEPS=4000,
    # MODEL CONFIG:
    D_MODEL=512,
    N_BLOCKS=6,
    N_HEADS=8,
    D_FF=2048,
    DROPOUT_PROBA=0.1,
    # OTHER:
    MODEL_SAVE_EPOCH_CNT=1000,
    DEVICE='cpu',
    LABEL_SMOOTHING=0.1,
)

