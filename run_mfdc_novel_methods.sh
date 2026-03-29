#!/usr/bin/env bash
# Run MFDC novel-method evaluation / fine-tuning on VOC.
# Usage: bash run_mfdc_novel_methods.sh <split_id> [method] [shots] [seeds] [run_mode]

set -euo pipefail

SPLIT_ID=${1:-}
METHOD=${2:-"pcb_fma pcb_fma_enhanced_neg neg_proto_guard"}
SHOTS=${3:-"1 5 10"}
SEEDS=${4:-"0"}
RUN_MODE=${5:-"infer_pretrained_novel"}

show_usage() {
    echo "Usage: bash run_mfdc_novel_methods.sh <split_id> [method] [shots] [seeds] [run_mode]"
    echo ""
    echo "Arguments:"
    echo "  split_id  : VOC split (1, 2, or 3)"
    echo "  method    : Method(s) to run, or 'all' (default: pcb_fma pcb_fma_enhanced_neg neg_proto_guard)"
    echo "  shots     : Shot settings (default: \"1 2 3 5 10\")"
    echo "  seeds     : Random seeds (default: \"0\")"
    echo "  run_mode  : finetune or infer_pretrained_novel (default: infer_pretrained_novel)"
    echo ""
    echo "Examples:"
    echo "  bash run_mfdc_novel_methods.sh 1"
    echo "  bash run_mfdc_novel_methods.sh 1 pcb_fma \"1 5\" \"0\""
    echo "  bash run_mfdc_novel_methods.sh 2 all \"1 2\" \"0 1\" infer_pretrained_novel"
}

if [ -z "${SPLIT_ID}" ]; then
    show_usage
    exit 1
fi

if [[ ! " ${SPLIT_ID} " =~ ^\ [123]\ $ ]]; then
    echo "Unsupported split_id: ${SPLIT_ID}"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

EXP_NAME=${EXP_NAME:-mfdc_novel_methods}
SAVE_DIR=${SAVE_DIR:-checkpoints/voc/${EXP_NAME}}
IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN_TORCH:-weight/resnet101-5d3b4d8f.pth}
BASE_EXP_NAME=${BASE_EXP_NAME:-vanilla_mfdc}
BASE_WEIGHT=${BASE_WEIGHT:-checkpoints/voc/${BASE_EXP_NAME}/base${SPLIT_ID}/model_reset_surgery.pth}
PRETRAINED_NOVEL_ROOT=${PRETRAINED_NOVEL_ROOT:-checkpoints/voc/${BASE_EXP_NAME}}

case "${RUN_MODE}" in
    finetune|train)
        RUN_MODE="finetune"
        RUN_MODE_DESC="fine-tune from base weights"
        RUN_MODE_DIR="finetune"
        ;;
    infer_pretrained_novel|pretrained_novel|eval)
        RUN_MODE="infer_pretrained_novel"
        RUN_MODE_DESC="eval-only from pretrained MFDC novel checkpoints"
        RUN_MODE_DIR="pretrained_eval"
        ;;
    *)
        echo "Unknown run_mode: ${RUN_MODE}"
        echo "Available run modes: finetune, infer_pretrained_novel"
        exit 1
        ;;
esac

declare -A METHOD_DIRS
METHOD_DIRS["pcb_fma"]="pcb_fma"
METHOD_DIRS["neg_proto_guard"]="neg_proto_guard"
METHOD_DIRS["pcb_fma_enhanced_neg"]="pcb_fma_enhanced_neg"

declare -A METHOD_SUFFIXES
METHOD_SUFFIXES["pcb_fma"]="pcb_fma"
METHOD_SUFFIXES["neg_proto_guard"]="neg_proto_guard"
METHOD_SUFFIXES["pcb_fma_enhanced_neg"]="pcb_fma_enhanced_neg"

if [ "${METHOD}" = "all" ]; then
    METHODS="pcb_fma pcb_fma_enhanced_neg neg_proto_guard"
else
    METHODS="${METHOD}"
fi

echo "=============================================="
echo "MFDC Novel Methods Runner"
echo "=============================================="
echo "Split: ${SPLIT_ID}"
echo "Methods: ${METHODS}"
echo "Shots: ${SHOTS}"
echo "Seeds: ${SEEDS}"
echo "Run mode: ${RUN_MODE} (${RUN_MODE_DESC})"
echo "Save dir: ${SAVE_DIR}/${RUN_MODE_DIR}"
echo "Base weight: ${BASE_WEIGHT}"
echo "Pretrained novel root: ${PRETRAINED_NOVEL_ROOT}"
echo "=============================================="

for method in ${METHODS}; do
    METHOD_DIR=${METHOD_DIRS[$method]:-}
    METHOD_SUFFIX=${METHOD_SUFFIXES[$method]:-}

    if [ -z "${METHOD_DIR}" ] || [ -z "${METHOD_SUFFIX}" ]; then
        echo "Unknown method: ${method}"
        echo "Available: pcb_fma, pcb_fma_enhanced_neg, neg_proto_guard"
        exit 1
    fi

    METHOD_SAVE_DIR=${SAVE_DIR}/${RUN_MODE_DIR}/${METHOD_DIR}/split${SPLIT_ID}
    mkdir -p "${METHOD_SAVE_DIR}"

    echo ""
    echo ">>> Running method: ${method}"

    for shot in ${SHOTS}; do
        for seed in ${SEEDS}; do
            echo ""
            echo "  Processing: ${shot}-shot, seed ${seed}"

            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting gfsod --split ${SPLIT_ID}

            TEMPLATE_CONFIG=configs/voc/novelMethods/${METHOD_DIR}/mfdc_gfsod_novelx_${shot}shot_seedx_${METHOD_SUFFIX}.yaml
            CONFIG_PATH=configs/voc/novelMethods/${METHOD_DIR}/mfdc_gfsod_novel${SPLIT_ID}_${shot}shot_seed${seed}_${METHOD_SUFFIX}.yaml

            if [ ! -f "${TEMPLATE_CONFIG}" ]; then
                echo "Missing template config: ${TEMPLATE_CONFIG}"
                exit 1
            fi

            cp "${TEMPLATE_CONFIG}" "${CONFIG_PATH}"
            sed -i "s/novelx/novel${SPLIT_ID}/g" "${CONFIG_PATH}"
            sed -i "s/seedx/seed${seed}/g" "${CONFIG_PATH}"

            OUTPUT_DIR=${METHOD_SAVE_DIR}/${shot}shot_seed${seed}

            if [ "${RUN_MODE}" = "infer_pretrained_novel" ]; then
                MODEL_WEIGHT=${PRETRAINED_NOVEL_ROOT}/mfdc_gfsod_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}/model_final.pth
                if [ ! -f "${MODEL_WEIGHT}" ]; then
                    echo "Missing pretrained novel checkpoint: ${MODEL_WEIGHT}"
                    exit 1
                fi
                python3 train_net.py --num-gpus 1 --eval-only --config-file "${CONFIG_PATH}" \
                    --opts MODEL.WEIGHTS "${MODEL_WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                           TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
            else
                if [ ! -f "${BASE_WEIGHT}" ]; then
                    echo "Missing base checkpoint: ${BASE_WEIGHT}"
                    exit 1
                fi
                python3 train_net.py --num-gpus 1 --config-file "${CONFIG_PATH}" \
                    --opts MODEL.WEIGHTS "${BASE_WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}" \
                           TEST.PCB_MODELPATH "${IMAGENET_PRETRAIN_TORCH}"
            fi

            rm -f "${CONFIG_PATH}"
            BASE_CONFIG_PATH=configs/voc/mfdc_gfsod_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            rm -f "${BASE_CONFIG_PATH}"

            echo "  Completed: ${shot}-shot, seed ${seed}"
        done
    done
done

echo ""
echo "=== All requested runs completed. ==="
