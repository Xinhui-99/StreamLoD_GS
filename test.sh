exp_name='output_meeting_exp_hard_threshold'
scenes=("discussion" "vrheadset" ) #"trimming"  "vrheadset" "discussion" "trimming"
dataset_path='/mnt/dongxu-fs2/data-ssd/xinhuiliu/dataset/DGS/MeetRoom'
 
removerates=(0.0)
grad_level_save=(1)
grad_threholds=(0.00000 0.00005 0.0001 0.0002 0.0003 0.0004 0.0005 0.001  0.002 0.003 0.004 0.005)
llffnumbers=(6 4) #23456711  3
epochs_rests=(10) #23456711
grad_threthold_modes=("hard_threshold")  #Gmm or hard_threshold
 
for grad_threhold in "${grad_threholds[@]}"
do
  for scene in "${scenes[@]}"
  do
    for llffnumber in "${llffnumbers[@]}"
    do
      for epochs_rest in "${epochs_rests[@]}"
      do
        python train.py --config configs/dynerf.yaml --log_ply -s $dataset_path/$scene -m $exp_name/output_sparse30/"${scene}3_r${grad_threhold}_llff${llffnumber}_eprest${epochs_rest}" --removerate $removerate --grad_threhold $grad_threhold --interval 30  --total_cameras 13 --epochs_rest $epochs_rest --llffnumber $llffnumber
      done
    done
  done
done



