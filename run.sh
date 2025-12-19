#!/bin/zsh
SEED=$((RANDOM % 10000))
LR=5e-5
HIDDEN_DIM=512
BATCH_SIZE=32
EPOCH_COUNT_FOR_VALID=1
NUM_EPOCHS=20
EMB_PROMPT="Summarize this entity record: "
CTPROMPT="normal"
EMBEDDING_TOOL="openai"
CT_LLM="gpt-4o-mini"
SLM="PATH_TO_SLM"
IS_CT="false"
COMPRESSION_STYLE="autoencoder"
ALPHA=0.4 
BETA=0.6
LAMDA=0.2
TRAIN_DATA_RATIO=1.0
WRONG_LABEL_RATIO=0.0
C=1
TASK="AG"
CONFIG_FILE="PATH_TO_CONFIG_FILE"
RESULT_DIR="PATH_TO_RESULT_DIR"
CHECKPOINT_DIR="PATH_TO_CHECKPOINT_DIR"

# Set CUDA device (optional, comment out to use all available GPUs)
# export CUDA_VISIBLE_DEVICES=0

python main.py --task $TASK \
    --config_file $CONFIG_FILE \
    --result_dir $RESULT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --wrong_label_ratio $WRONG_LABEL_RATIO \
    --C $C \
    --is_ct $IS_CT \
    --train_data_ratio $TRAIN_DATA_RATIO \
    --seed $SEED \
    --embedding_tool $EMBEDDING_TOOL \
    --slm $SLM \
    --emb_prompt "$EMB_PROMPT" \
    --ct_prompt "$CTPROMPT" \
    --ct_llm "$CT_LLM" \
    --compression_style "$COMPRESSION_STYLE" \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --batch_size $BATCH_SIZE \
    --epoch_count_for_valid $EPOCH_COUNT_FOR_VALID \
    --num_epochs $NUM_EPOCHS \
    --alpha $ALPHA \
    --beta $BETA \
    --lamda $LAMDA