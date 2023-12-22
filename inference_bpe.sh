python3 bin/inference_single_pair_visuals.py  --data_dir "../SARA_released/" \
  --model_path "./logdir/exp_bpe/model/model_epoch39.pth" \
  --video1 "../../Dataset_CVDLPT_Videos_Segments_11_2023/E0_P0_T0_C0_seg0.mp4" \
  --vid1_json_dir "../E0_P0_T0_C0_seg0_2D.json" \
  --video2 "../../Dataset_CVDLPT_Videos_Segments_11_2023/E0_P0_T0_C0_seg1.mp4" \
  --vid2_json_dir "../E0_P0_T0_C0_seg1_2D.json" \
  -h1 400 \
  -h2 400 \
  -w1 300 \
  -w2 300 \
  --use_flipped_motion \
  --video_sampling_window_size 20 \
  --video_sampling_stride 1 \
  --similarity_measurement_window_size 5 \
  --out_filename output.mp4 \
  --thresh 0.4


#--vid1_json_dir "../NTU_motion_sim_annotations/sample.json" \
#  --vid1_json_dir "../E0_P0_T0_C0_seg0_2D.json"
