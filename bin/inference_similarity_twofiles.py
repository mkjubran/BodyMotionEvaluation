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
import itertools
from tslearn.metrics import dtw_path
import torch
from dtaidistance import dtw_ndim
import json

#file1="E0_P0_T0_C2_seg5_2D_emb"
#file2="E0_P1_T0_C2_seg6_2D_emb"

body_parts_name = ['ra', 'la', 'rl', 'll', 'torso']

PerWindowsSize = False
PerBodyPart = False

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if len(chunk) != size:
            break
        yield chunk

def get_similarity(seq1_features, seq2_features):
        path, dist = dtw_path(np.array(seq1_features), np.array(seq2_features))
        similarities_per_path = []
        for i in range(len(path)):
            cosine_sim = cosine_score(torch.Tensor(seq1_features[path[i][0]]),
                                           torch.Tensor(seq2_features[path[i][1]])).numpy()
            similarities_per_path.append(cosine_sim)
        total_path_similarity = sum(similarities_per_path) / len(path)
        return total_path_similarity

def get_similarity_wpath(seq1_features, seq2_features, path):
        similarities_per_path = []
        for i in range(len(path)):
            cosine_sim = cosine_score(torch.Tensor(seq1_features[path[i][0]]),
                                           torch.Tensor(seq2_features[path[i][1]])).numpy()
            similarities_per_path.append(cosine_sim)
        total_path_similarity = sum(similarities_per_path) / len(path)
        return total_path_similarity

cosine_score = torch.nn.CosineSimilarity(dim=0, eps=1e-50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="", required=True, help="path to dataset dir")
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
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
    parser.add_argument('--embeddings_path', type=str, help="path for all npz files")
    parser.add_argument('--mp4_path', type=str, help="path for all mp4 files")
    parser.add_argument('--file1emb', type=str, help="path for file1 embeddings")
    parser.add_argument('--file2emb', type=str, help="path for file2 embeddings")
    args = parser.parse_args()

    file1=args.file1emb
    file2=args.file2emb

    # load meanpose and stdpose
    mean_pose_bpe = np.load(os.path.join(args.data_dir, 'meanpose_rc_with_view_unit64.npy'))
    std_pose_bpe = np.load(os.path.join(args.data_dir, 'stdpose_rc_with_view_unit64.npy'))

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    config = Config(args)

    # Get the list of files in the folder
    file_list = [file.split('.')[0] for file in os.listdir(args.embeddings_path) if (os.path.isfile(os.path.join(args.embeddings_path, file)) and (('seg2' in file) or ('seg6' in file)))]
    #file_list = [file.split('.')[0] for file in os.listdir(args.embeddings_path) if (os.path.isfile(os.path.join(args.embeddings_path, file)))]

    # loop through all files and store them in the dictionary
    cnt_file1 = 0
    len_file_list = len(file_list)
    if True:
           file1_score=[]
           file2_meat_score=[]
           #file1="E4_P0_T1_C2_seg6_2D_emb"
           #file2="E4_P0_T1_C2_seg6_2D_emb"
           data_file1=np.load(os.path.join(args.embeddings_path, f"{file1}.npz"))['data']
           data_file2=np.load(os.path.join(args.embeddings_path, f"{file2}.npz"))['data']
           print(file1, file2)
           if (not PerWindowsSize) and (not PerBodyPart):
              # Evaluate the exercise quality for all frames and all bodey parts, single score will be assigned
              file2_meta_score=[file2.split('_')[0][1:],file2.split('_')[1][1:],file2.split('_')[2][1:],file2.split('_')[3][1:],file2.split('_')[4][3:]]
              similarity_score=[]
              seq1_features_vector = data_file1
              seq2_features_vector = data_file2
              path = dtw_ndim.warping_path(seq1_features_vector, seq2_features_vector)
              similarity_score = get_similarity_wpath(seq1_features_vector, seq2_features_vector, path)
              file2_meta_score.append(similarity_score)
              file1_score.append(file2_meta_score)
           elif (not PerWindowsSize) and (PerBodyPart):
              # Evaluate the exercise quality for all frames, multiple scores will be assigned for the same exercise considering different body parts
              file2_meta_score=[file2.split('_')[0][1:],file2.split('_')[1][1:],file2.split('_')[2][1:],file2.split('_')[3][1:],file2.split('_')[4][3:]]
              similarity_score=[]
              similarity_score_by_body_part=[]
              for cntbp in range(5):
                    if cntbp != 4:
                       seq1_features_vector = data_file1[:,(cntbp*128):((cntbp+1)*128)]
                       seq2_features_vector = data_file2[:,(cntbp*128):((cntbp+1)*128)]
                    else:
                       seq1_features_vector = data_file1[:,(cntbp*128):]
                       seq2_features_vector = data_file2[:,(cntbp*128):]
                    path = dtw_ndim.warping_path(seq1_features_vector, seq2_features_vector)
                    similarity_score_by_body_part.append(get_similarity_wpath(seq1_features_vector, seq2_features_vector, path))
              file2_meta_score.extend(similarity_score_by_body_part)
              file1_score.append(file2_meta_score)
           elif (PerWindowsSize) and (PerBodyPart):
              # Evaluate the exercise quality based on a similarity window size.
              # Multiple scores will be assigned for the same exercise, considering different body parts and exercise segments.
              file2_meta_score=[file2]
              seq1_features=[]
              for cntF in range(data_file1.shape[0]):
                 seq1_features_vector=[]
                 for cntbp in range(5):
                    if cntbp != 4:
                       seq1_features_vector.append(data_file1[cntF,(cntbp*128):((cntbp+1)*128)])
                    else:
                       seq1_features_vector.append(data_file1[cntF,(cntbp*128):])
                 seq1_features.append(seq1_features_vector)

              seq2_features=[]
              for cntF in range(data_file2.shape[0]):
                 seq2_features_vector=[]
                 for cntbp in range(5):
                    if cntbp != 4:
                       seq2_features_vector.append(data_file2[cntF,(cntbp*128):((cntbp+1)*128)])
                    else:
                       seq2_features_vector.append(data_file2[cntF,(cntbp*128):])
                 seq2_features.append(seq2_features_vector)

              similarity_score_per_window = []
              similarity_window_size=args.similarity_measurement_window_size
              for subseq1_features, subseq2_features in zip(*(chunked_iterable(seq1_features, similarity_window_size),
                                                              chunked_iterable(seq2_features, similarity_window_size))):

                 assert len(subseq1_features) == len(subseq2_features) and len(subseq2_features) != 0

                 similarity_score_by_body_part = {}
                 for bp_idx, bp in enumerate(body_parts_name):
                      subseq1_features_bp = [subseq1_features[subseq_temporal_idx][bp_idx] for subseq_temporal_idx in
                                             range(len(subseq1_features))]
                      subseq2_features_bp = [subseq2_features[subseq_temporal_idx][bp_idx] for subseq_temporal_idx in
                                             range(len(subseq2_features))]
                      similarity_score_by_body_part[bp] = get_similarity(subseq1_features_bp, subseq2_features_bp)

                 similarity_score_per_window.append(similarity_score_by_body_part)
              file2_meta_score.append(similarity_score_per_window)
              file1_score.append(file2_meta_score)

           print(file1_score)
