python train_bart_model.py --dataset_name irishman --random_masking --prob 0.3 --enc_dec_model facebook/bart-base --learning_rate 1e-4 --lr_warmup_steps 2000 --train_batch_size 256 --sample_every 2000 --save_every 10000 --l2_normalize_latents

