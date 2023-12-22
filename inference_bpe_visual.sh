file1="E6_P8_T0_C0_seg0"
file2="E6_P7_T0_C0_seg0"

python3 bin/inference_single_pair_visuals.py  --data_dir "../SARA_released/" \
  --model_path "./logdir/exp_bpe/model/model_epoch70.pth" \
  --video1 "../../Dataset_CVDLPT_Videos_Segments_11_2023/"$file1".mp4" \
  --vid1_json_dir "../../Dataset_CVDLPT_Videos_Segments_npz_11_2023/"$file1"_2D.npz" \
  --video2 "../../Dataset_CVDLPT_Videos_Segments_11_2023/"$file2".mp4" \
  --vid2_json_dir "../../Dataset_CVDLPT_Videos_Segments_npz_11_2023/"$file2"_2D.npz" \
  -h1 400 \
  -h2 400 \
  -w1 300 \
  -w2 300 \
  --use_flipped_motion \
  --video_sampling_window_size 20 \
  --video_sampling_stride 1 \
  --similarity_measurement_window_size 20 \
  --out_filename output.mp4 \
  --thresh 0.9

