file1="E6_P8_T0_C0_seg0"
file2="E6_P7_T0_C0_seg0"


python3 bin/inference_embeddings.py  --data_dir "../SARA_released/" \
  --model_path "./logdir/exp_bpe/model/model_epoch70.pth" \
  --pose_detection "MoveNet" \
  -h1 400 \
  -w1 300 \
  --video_sampling_window_size 13 \
  --video_sampling_stride 1 \
  --similarity_measurement_window_size 1 \
  --out_filename output.mp4 \
  --thresh 0.9 \
  --pose_detection "MoveNet" \
  --npz_path "../../Dataset_CVDLPT_Videos_Segments_MoveNet_thunder_npz" \
  --mp4_path "../../Dataset_CVDLPT_Videos_Segments_11_2023" 


  #--use_flipped_motion \
