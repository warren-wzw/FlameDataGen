#!/bin/bash

#setup env
# ENV_NAME='XXXX'
# conda create --name $ENV_NAME python==3.8.0
#source /home/warren/anaconda3/bin/activate  /home/warren/anaconda3/envs/$ENV_NAME
#pip install -r requirment.txt

#create folders
OUTPUT_PATH="./output"
OUTPUT_PATH_TF="./output/tflog"
OUTPUT_PATH_DATA="./output/output_images"
OUTPUT_PATH_MODEL="./output/output_models"
DATASET_PATH="./dataset"
DATASET_PATH_CACHE="./dataset/cache"
DIR_PATH_SRC="./dataset/src"
DIR_PATH_SRC_TEST="./dataset/src/test"
DIR_PATH_SRC_TRAIN="./dataset/src/train"
DIR_PATH_SRC_VAL="./dataset/src/val"
DIR_PATH_LABEL="./dataset/label"
DIR_PATH_LABEL_TEST="./dataset/label/test"
DIR_PATH_LABEL_TRAIN="./dataset/label/train"
DIR_PATH_LABEL_VAL="./dataset/label/val"

create_directory() {
    local DIR_PATH=$1

    if [ -d "$DIR_PATH" ]; then
        echo "文件夹 '$DIR_PATH' 已经存在"
    else
        # 创建文件夹
        mkdir -p "$DIR_PATH"
        if [ $? -eq 0 ]; then
            echo "文件夹 '$DIR_PATH' 创建成功"
        else
            echo "创建文件夹 '$DIR_PATH' 时出错"
        fi
    fi
}

create_directory "$OUTPUT_PATH"
create_directory "$OUTPUT_PATH_TF"
create_directory "$OUTPUT_PATH_DATA"
create_directory "$OUTPUT_PATH_MODEL"
create_directory "$DATASET_PATH"
create_directory "$DATASET_PATH_CACHE"
create_directory "$DIR_PATH_SRC"
create_directory "$DIR_PATH_SRC_TEST"
create_directory "$DIR_PATH_SRC_TRAIN"
create_directory "$DIR_PATH_SRC_VAL"
create_directory "$DIR_PATH_LABEL"
create_directory "$DIR_PATH_LABEL_TEST"
create_directory "$DIR_PATH_LABEL_TRAIN"
create_directory "$DIR_PATH_LABEL_VAL"
