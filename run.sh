
exp_name='/mnt/dongxu-fs2/data-ssd/xinhuiliu/demo/3DGStream/StreamLoD/StreamLoD1101_all/output_dynerf_exp2'
scenes=("cut_roasted_beef" “flame_salmon_1”) #
dataset_path='/mnt/dongxu-fs2/data-ssd/xinhuiliu/dataset/DGS/dynerf'
 
 
removerates=(0.0)
grad_level_save=(1)
grad_threholds=(0.00000)
llffnumbers=(6) #23456711
epochs_rests=(6) #23456711
grad_threthold_modes=("Gmm")  #Gmm or hard_threshold
 
for removerate in "${removerates[@]}"
do
  for scene in "${scenes[@]}"
  do
    for llffnumber in "${llffnumbers[@]}"
    do
      for epochs_rest in "${epochs_rests[@]}"
      do
        echo "Training on $scene with removerate $removerate..."
        python train.py --config configs/dynerf.yaml --log_ply -s $dataset_path/$scene -m $exp_name/output_sparse30/"${scene}3_r${removerate}_llff${llffnumber}_eprest${epochs_rest}" --removerate $removerate --interval 30  --total_cameras 21 --epochs_rest $epochs_rest --llffnumber $llffnumber
      done
    done
  done
done
 
 