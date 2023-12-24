import numpy as np # linear algebra
from dtaidistance import dtw_ndim
import cv2
import sys
import pdb

def ProcessData(Ex1_npz,PoseDetection='GAST'):
    if PoseDetection == 'GAST':
       Ex1 = np.load(Ex1_npz)['reconstruction'][0]
    elif PoseDetection == 'MoveNet':
       Ex1 = np.load(Ex1_npz)['arr_0']
       Ex1[:,:,0] = Ex1[:,:,0]*1080;Ex1[:,:,1] = Ex1[:,:,1]*1920
       Ex1[Ex1 == -1] = 0;
    else:
       print("No DTW because Pose detection method in unknown")
       return
    np.savez_compressed('./DTWOutputFiles/Ex1_DTW.npz', reconstruction=Ex1.reshape(1,Ex1.shape[0],Ex1.shape[1],Ex1.shape[2]))



def DTW(Ex1_npz, Ex1_mp4, Ex2_npz, Ex2_mp4, PoseDetection='GAST', visualize=True):

    if PoseDetection == 'GAST':
       Ex1 = np.load(Ex1_npz)['reconstruction'][0]
       Ex2 = np.load(Ex2_npz)['reconstruction'][0]

    elif PoseDetection == 'MoveNet':
       Ex1 = np.load(Ex1_npz)['arr_0']
       Ex2 = np.load(Ex2_npz)['arr_0']
       Ex1[:,:,0] = Ex1[:,:,0]*1080;Ex1[:,:,1] = Ex1[:,:,1]*1920
       Ex2[:,:,0] = Ex2[:,:,0]*1080;Ex2[:,:,1] = Ex2[:,:,1]*1920
       Ex1[Ex1 == -1] = 0;Ex2[Ex2 == -1] = 0;

       # only for testing the system without DTW made by jubran, must be commented - 2 lines
       np.savez_compressed('./DTWOutputFiles/Ex1_DTW.npz', reconstruction=Ex1.reshape(1,Ex1.shape[0],Ex1.shape[1],Ex1.shape[2]))
       np.savez_compressed('./DTWOutputFiles/Ex2_DTW.npz', reconstruction=Ex2.reshape(1,Ex2.shape[0],Ex2.shape[1],Ex2.shape[2]))
       return
    else:
       print("No DTW because Pose detection method in unknown")
       return

    x = Ex1.reshape(Ex1.shape[0],-1)
    y = Ex2.reshape(Ex2.shape[0],-1)

    d = dtw_ndim.distance(x, y)
    path = dtw_ndim.warping_path(x, y)

    Ex1_DTW=[]
    Ex2_DTW=[]
    Ex1_frame_order=[]
    Ex2_frame_order=[]
    for (i,j) in path:
       Ex1_DTW.append(Ex1[i,:,:])
       Ex2_DTW.append(Ex2[j,:,:])

       Ex1_frame_order.append(i)
       Ex2_frame_order.append(j)

    Ex1_DTW=np.stack( Ex1_DTW, axis=0)
    Ex2_DTW=np.stack( Ex2_DTW, axis=0)

    Ex1_DTW=Ex1_DTW.reshape(1,Ex1_DTW.shape[0],Ex1_DTW.shape[1],Ex1_DTW.shape[2])
    Ex2_DTW=Ex2_DTW.reshape(1,Ex2_DTW.shape[0],Ex2_DTW.shape[1],Ex2_DTW.shape[2])

    np.savez_compressed('./DTWOutputFiles/Ex1_DTW.npz', reconstruction=Ex1_DTW)
    np.savez_compressed('./DTWOutputFiles/Ex2_DTW.npz', reconstruction=Ex2_DTW)

    if visualize:
       reorder_mp4_frames(Ex1_mp4, './DTWOutputFiles/Ex1_DTW.mp4', Ex1_frame_order)
       reorder_mp4_frames(Ex2_mp4, './DTWOutputFiles/Ex2_DTW.mp4', Ex2_frame_order)

    return


def reorder_mp4_frames(input_file, output_file, frame_order):
    # Open the input video file
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frames_per_second = cap.get(5)

    # Create VideoWriter object to save the reordered frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frames_per_second, (frame_width, frame_height))

    # Read frames
    frames=[]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break   
        frames.append(frame)

    for i in frame_order:
      if i <= len(frames):
        # Write the reordered frame to the output video file
        out.write(frames[i])

    # Release video capture and writer objects
    cap.release()
    out.release()

    return

