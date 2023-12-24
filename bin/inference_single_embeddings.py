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
from bpe.PTLoadandDTW import ProcessData #jubran
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('--video1', type=str, required=True, help="video1 mp4 path", default=None)
    parser.add_argument('-v1', '--vid1_json_dir', type=str, required=True, help="video1's coco annotation json")
    parser.add_argument('-h1', '--img1_height', type=int, help="video1's height", default=480)
    parser.add_argument('-w1', '--img1_width', type=int, help="video1's width", default=854)
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

    similarity_analyzer = SimilarityAnalyzer(config, args.model_path)

    # Get the list of files in the folder
    #file_list = [file.split('.')[0] for file in os.listdir(args.npz_path) if (os.path.isfile(os.path.join(args.npz_path, file)) and ('seg1' in file))]
    file_list = [file.split('.')[0] for file in os.listdir(args.npz_path) if (os.path.isfile(os.path.join(args.npz_path, file)))]

    # loop through all files and store them in the dictionary
    for file1 in tqdm(file_list, desc=f"Extract Features"):
         file_embeddings_npz=f"./output/{file1}_emb.npz"
         if not os.path.exists(file_embeddings_npz):

           #print("Apply DTW to seq1 and seq2")
           vid1_npz=os.path.join(args.npz_path,f"{file1}.npz")
           vid1_mp4=os.path.join(args.mp4_path,f"{file1[0:-3]}.mp4")

           ProcessData(vid1_npz, PoseDetection=args.pose_detection) #jubran

           args.vid1_json_dir = './DTWOutputFiles/Ex1_DTW.npz'

           # for NTU-RGB test - it used w:1920, h:1080
           h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)

           # get input suitable for motion similarity analyzer
           seq1 = cocopose2motion(config.unique_nr_joints, args.vid1_json_dir, scale=scale1,
                           visibility=args.use_invisibility_aug)

           # TODO: interpoloation or oef filtering for missing poses.
           seq1 = preprocess_sequence(seq1)

           seq1_origin = preprocess_motion2d_rc(seq1, mean_pose_bpe, std_pose_bpe,
                                         invisibility_augmentation=args.use_invisibility_aug,
                                         use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)

           # move input to device
           seq1_origin = seq1_origin.to(config.device)

           # get embeddings
           seq1_features = similarity_analyzer.get_embeddings(seq1_origin, video_window_size=args.video_sampling_window_size,
                                                       video_stride=args.video_sampling_stride)

           # To save embeddings - Jubran
           seq1_features_np=[]
           for cntF in range(len(seq1_features)):
              seq1_features_vector=[]
              for cntbp in range(5):
                  seq1_features_vector.extend(seq1_features[cntF][cntbp])
              seq1_features_np.append(seq1_features_vector)
           seq1_features_np_stack = np.stack(seq1_features_np,axis=0)

           # Save the array to a .npz file
           np.savez(file_embeddings_npz, data=seq1_features_np_stack)

