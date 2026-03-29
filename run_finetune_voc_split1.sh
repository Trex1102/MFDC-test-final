#!/usr/bin/env bash

EXP_NAME=vanilla_mfdc
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN_TORCH=weight/resnet101-5d3b4d8f.pth
SPLIT_ID=1
BASE_WEIGHT=${SAVE_DIR}/base${SPLIT_ID}/model_reset_surgery.pth

echo "=== Using base weight: ${BASE_WEIGHT} ==="
echo "=== Starting fine-tuning for VOC split ${SPLIT_ID} ==="

for seed in 0
do
    for shot in 2 3
    do
        echo ""
        echo ">>> Running ${shot}-shot seed ${seed} ..."
        python3 tools/create_config.py --dataset voc --config_root configs/voc \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/mfdc_gfsod_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/mfdc_gfsod_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 train_net.py --num-gpus 1 --config-file ${CONFIG_PATH} \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm -f ${CONFIG_PATH}
        echo ">>> Done ${shot}-shot seed ${seed}"
    done
done

echo ""
echo "=== All fine-tuning complete! ==="
