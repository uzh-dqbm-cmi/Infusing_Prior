python3 -m src.main_R2Gen \
        --image_dir /data/iu_xray/images/ \
        --ann_path /data/iu_xray/new_annotation.json \
        --ann_path_tokenizer /data/iu_xray/new_annotation.json \
        --vocab_path /data/iu_xray/vocab.pickle \
        --dataset_name iu_xray \
        --src_max_seq_length 60 \
        --threshold 3 \
        --batch_size 16 \
        --epochs 100 \
        --save_dir /results/iu_xray/  \
        --record_dir /results/iu_xray/  \
        --step_size 50 \
        --gamma 0.1 \
        --report_mode report \
        --seed $SEED \
        --n_gpu 1 \
        --labeled_report_path /data/labeled_iu_reports.csv \
        --infuse_prior true \

