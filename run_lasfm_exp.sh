nohup python Speed_main_exp.py \
      --dataset last-fm \
      --lr 0.001 \
      --context_hops 2 \
      --num_neg_sample 600 \
      --margin 0.7 \
      --max_iter 2 \
      --test_batch_size 2048 \
      --lr_dc_step 6 \
      --lr_dc 0.5 \
      --gpu_id 2 > ./result/lasfm_speed_exp_tran_bs512.log 2>&1 &