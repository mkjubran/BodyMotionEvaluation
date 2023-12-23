import os
import argparse

import pdb

import numpy as np
import math

from bpe import Config
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.motion import preprocess_motion2d_rc, cocopose2motion
from bpe.functional.utils import pad_to_height
from bpe.functional.visualization import preprocess_sequence, video_out_with_imageio

from dtaidistance import dtw_ndim #jubran
from bpe.PTLoadandDTW import DTW #jubran
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('--video1', type=str, required=True, help="video1 mp4 path", default=None)
    parser.add_argument('--video2', type=str, required=True, help="video2 mp4 path", default=None)
    parser.add_argument('-v1', '--vid1_json_dir', type=str, required=True, help="video1's coco annotation json")
    parser.add_argument('-v2', '--vid2_json_dir', type=str, required=True, help="video2's coco annotation json")
    parser.add_argument('-h1', '--img1_height', type=int, help="video1's height", default=480)
    parser.add_argument('-w1', '--img1_width', type=int, help="video1's width", default=854)
    parser.add_argument('-h2', '--img2_height', type=int, help="video2's height", default=480)
    parser.add_argument('-w2', '--img2_width', type=int, help="video2's width", default=854)
    parser.add_argument('-pad2', '--pad2', type=int, help="video2's start frame padding", default=0)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--out_path', type=str, default='./visual_results', required=False)
    parser.add_argument('--out_filename', type=str, default='twice.mp4', required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
   # related to video processing
    parser.add_argument('--video_sampling_window_size', type=int, default=16,
                        help='window size to use for similarity prediction')
    parser.add_argument('--video_sampling_stride', type=int, default=16,
                        help='stride determining when to start next window of frames')
    parser.add_argument('--use_all_joints_on_each_bp', action='store_true',
                        help="using all joints on each body part as input, as opposed to particular body part")

    parser.add_argument('--similarity_measurement_window_size', type=int, default=1,
                        help='measuring similarity over # of oversampled video sequences')
    parser.add_argument('--similarity_distance_metric', choices=["cosine", "l2"], default="cosine")
    parser.add_argument('--privacy_on', action='store_true',
                        help='when on, no original video or sound in present in the output video')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold to seprate positive and negative classes')
    parser.add_argument('--connected_joints', action='store_true', help='connect joints with lines in the output video')
   
    #Added by jubran
    parser.add_argument('--pose_detection', type=str, help="GAST, MoveNet")
    parser.add_argument('--npz_path', type=str, help="path for all npz files")
    parser.add_argument('--mp4_path', type=str, help="path for all mp4 files")
    args = parser.parse_args()

    # load meanpose and stdpose
    mean_pose_bpe = np.load(os.path.join(args.data_dir, 'meanpose_rc_with_view_unit64.npy'))
    std_pose_bpe = np.load(os.path.join(args.data_dir, 'stdpose_rc_with_view_unit64.npy'))

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    config = Config(args)

    #pdb.set_trace()

    similarity_analyzer = SimilarityAnalyzer(config, args.model_path)

    # Get the list of files in the folder
    file_list = [file.split('.')[0] for file in os.listdir(args.npz_path) if (os.path.isfile(os.path.join(args.npz_path, file)) and ('seg1' in file))]

    # loop through all files and store them in the dictionary
    cnt_file1=0
    for file1 in file_list:
       cnt_file1 += 1; cnt_file2=0
       for file2 in tqdm(file_list, desc=f"[{cnt_file1}/{len(file_list)}]:{file1}"):
       #for file2 in file_list:
         motion_similarity_npz_file=f"./output/{file1}_{file2}.npz"
         cnt_file2 += 1
         if not os.path.exists(motion_similarity_npz_file):

           #print(f"[{cnt_file1}/{len(file_list)}]:{file1}, [{cnt_file2}/{len(file_list)}]:{file2}")

           #print("Apply DTW to seq1 and seq2")
           vid1_npz=os.path.join(args.npz_path,f"{file1}.npz")
           vid1_mp4=os.path.join(args.mp4_path,f"{file1[0:-3]}.mp4")
           vid2_npz=os.path.join(args.npz_path,f"{file2}.npz")
           vid2_mp4=os.path.join(args.mp4_path,f"{file2[0:-3]}.mp4")
           DTW(vid1_npz,vid1_mp4,vid2_npz,vid2_mp4, PoseDetection=args.pose_detection , visualize=False) #jubran

           args.vid1_json_dir = './DTWOutputFiles/Ex1_DTW.npz'
           args.video1= './DTWOutputFiles/Ex1_DTW.mp4'
           args.vid2_json_dir = './DTWOutputFiles/Ex2_DTW.npz'
           args.video2 = './DTWOutputFiles/Ex2_DTW.mp4'

           # for NTU-RGB test - it used w:1920, h:1080
           h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)
           h2, w2, scale2 = pad_to_height(config.img_size[0], args.img2_height, args.img2_width)

           # get input suitable for motion similarity analyzer
           seq1 = cocopose2motion(config.unique_nr_joints, args.vid1_json_dir, scale=scale1,
                           visibility=args.use_invisibility_aug)
           seq2 = cocopose2motion(config.unique_nr_joints, args.vid2_json_dir, scale=scale2,
                           visibility=args.use_invisibility_aug)[:, :, args.pad2:]

           # TODO: interpoloation or oef filtering for missing poses.
           seq1 = preprocess_sequence(seq1)
           seq2 = preprocess_sequence(seq2)

           seq1_origin = preprocess_motion2d_rc(seq1, mean_pose_bpe, std_pose_bpe,
                                         invisibility_augmentation=args.use_invisibility_aug,
                                         use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
           seq2_origin = preprocess_motion2d_rc(seq2, mean_pose_bpe, std_pose_bpe,
                                         invisibility_augmentation=args.use_invisibility_aug,
                                         use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)

           # move input to device
           seq1_origin = seq1_origin.to(config.device)
           seq2_origin = seq2_origin.to(config.device)

           #pdb.set_trace()

           # get embeddings
           seq1_features = similarity_analyzer.get_embeddings(seq1_origin, video_window_size=args.video_sampling_window_size,
                                                       video_stride=args.video_sampling_stride)
           seq2_features = similarity_analyzer.get_embeddings(seq2_origin, video_window_size=args.video_sampling_window_size,
                                                       video_stride=args.video_sampling_stride)
           # get motion similarity
           motion_similarity_per_window = \
                  similarity_analyzer.get_similarity_score(seq1_features, seq2_features,
                                                 similarity_window_size=args.similarity_measurement_window_size)
           if args.use_flipped_motion:
                  seq1_flipped = preprocess_motion2d_rc(seq1, mean_pose_bpe, std_pose_bpe, flip=args.use_flipped_motion,
                                              invisibility_augmentation=args.use_invisibility_aug,
                                              use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
                  seq1_flipped = seq1_flipped.to(config.device)
                  seq1_flipped_features = similarity_analyzer.get_embeddings(seq1_flipped,
                                                                   video_window_size=args.video_sampling_window_size,
                                                                   video_stride=args.video_sampling_stride)
                  motion_similarity_per_window_flipped = \
                          similarity_analyzer.get_similarity_score(seq1_flipped_features, seq2_features,
                                                     similarity_window_size=args.similarity_measurement_window_size)
                  for temporal_idx in range(len(motion_similarity_per_window)):
                      for key in motion_similarity_per_window[temporal_idx].keys():
                          motion_similarity_per_window[temporal_idx][key] = max(motion_similarity_per_window[temporal_idx][key],
                                                                      motion_similarity_per_window_flipped[
                                                                          temporal_idx][key])


           #by Jubran
           #pdb.set_trace()

           motion_similarity_per_frame=[] #meaningfull  when window_size = 1
           fraction = len(motion_similarity_per_window)/seq1[1].shape[1]
           for cnt in range(seq1[1].shape[1]):
               temporal_idx = math.floor(cnt*fraction)

               ra=motion_similarity_per_window[temporal_idx]['ra']
               la=motion_similarity_per_window[temporal_idx]['la']
               rl=motion_similarity_per_window[temporal_idx]['rl']
               ll=motion_similarity_per_window[temporal_idx]['ll']
               torso=motion_similarity_per_window[temporal_idx]['torso']

               motion_similarity_per_frame.append([cnt,ra,la,rl,ll,torso])
               motion_similarity_per_frame_npz = np.stack(motion_similarity_per_frame,axis=0)

           # Save the array to a .npz file
           np.savez(motion_similarity_npz_file, data=motion_similarity_per_frame_npz)

           if False:
              # suppose same video horizontal
              video_width = int(config.img_size[0] / args.img1_height * args.img1_width + config.img_size[0] / args.img2_height * args.img2_width)
              video_height = config.img_size[0]

              video_out_with_imageio(output_path=os.path.join(args.out_path, args.out_filename),
                           width=video_width, height=video_height,
                           sequence1=seq1.transpose(2, 0, 1), sequence2=seq2.transpose(2, 0, 1),
                           video1_path=args.video1, video2_path=args.video2,
                           left_padding=int(config.img_size[0] / args.img1_height * args.img1_width),
                           pad2=args.pad2,
                           motion_similarity_per_window=motion_similarity_per_window, is_debug=args.debug,
                           thresh=args.thresh,
                           privacy_on=args.privacy_on, is_connected_joints=args.connected_joints)



