#!/bin/bash
python eval.py \
    --model_name  "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/Xkev/Llama-3.2V-11B-cot"\
    --parquet_files "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/eval/refcoco/refer/data/refcoco/testA-00000-of-00001.parquet" \
    --result_file "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/eval/refcoco/result_files/refcoco/testA/tesA_cot.csv"
    # --finetuning_path "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/llama/llama-cookbook/lora/model_ep1"




    # "/mnt/pfs-mc0p4k/nlu/team/yuhaofu/modle_weight/Llama-3.2-11B-Vision-Instruct"