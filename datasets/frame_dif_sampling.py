
import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve2d

def laplacian_pyramid(image, n_levels=6):
    """Build Laplacian pyramid using Burt & Adelson filter."""
    f = np.array([.05, .25, .4, .25, .05])
    filt = np.outer(f, f)
    pyr = []
    J = image.copy()
    for _ in range(n_levels - 1):
        # Low-pass filter and downsample
        low = convolve(J, filt, mode='reflect')[::2, ::2]
        # Upsample and interpolate
        up = np.zeros_like(J)
        up[::2, ::2] = low * 4
        up = convolve(up, filt, mode='reflect')
        pyr.append(J - up)
        J = low
    pyr.append(J)
    return pyr

def DN_filters():
    """Default DN parameters for 6 levels."""
    sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
    F2_list = []
    # Level 1 kernel (from MATLAB)
    F2_list.append(np.array([
        [0,0,0,0,0],
        [0,0,0.1011,0,0],
        [0,0.1493,0,0.1460,0.0072],
        [0,0,0.1015,0,0],
        [0,0,0,0,0]
    ]))
    # Fill remaining kernels with zeros or replicate pattern
    for _ in range(5):
        F2_list.append(np.zeros((5,5)))
    return [{'sigma': s, 'F2': F2} for s, F2 in zip(sigmas, F2_list)]

def NLP_transform(image, dn_filts):
    """Apply Laplacian pyramid + advanced DN logic."""
    lap_pyr = laplacian_pyramid(image, n_levels=len(dn_filts))
    dn_pyr = []
    for level, band in enumerate(lap_pyr):
        A2 = convolve2d(np.abs(band), dn_filts[level]['F2'], mode='same')
        dn = band / (dn_filts[level]['sigma'] + A2)
        nn = band / (dn_filts[level]['sigma'] + A2)
        # # Extra nonlinear steps
        # dn3 = convolve2d(np.abs(nn), dn_filts[level]['F2'], mode='same')
        # dn4 = np.maximum(dn3, 0)  # subplus equivalent
        # final = nn / (dn4 + 1e-6)  # avoid division by zero
        # final2 = np.log1p(np.exp(final))  # log(1+exp(x))
        dn_pyr.append(dn)
    return dn_pyr, lap_pyr

def frame_LGN_features(image, dn_filts=None):
    """Compute LGN-like features for a single frame."""
    if dn_filts is None:
        dn_filts = DN_filters()
    Y_ori, Lap_ori = NLP_transform(image, dn_filts)
    return Y_ori, Lap_ori

from decord import VideoReader, cpu
from sklearn.decomposition import PCA
def frame_diff_sampling_video(video_path, max_frames=None, num_frames=16):
   
    if "360p" not in video_path:
        vr = VideoReader(video_path, ctx=cpu(), width=640, height=360)
    else:
        vr = VideoReader(video_path, ctx=cpu()) 
    all_features = []
    frame_count = len(vr) if max_frames is None else min(max_frames, len(vr))
    diff_pre = vr[:frame_count-1].asnumpy().astype(np.float32) - vr[1:frame_count].asnumpy().astype(np.float32)
    # MSE per frame
    mse_per_frame= np.mean(diff_pre ** 2, axis=(1,2,3))  # shape: (frame_count-1,)
    m = [(mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame)-2)]
    # m = mse_per_frame[i] if i == 0 or i == len(mse_per_frame)-1 else (mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame))
    m = [mse_per_frame[0]] + m + [mse_per_frame[-1]]
    m = np.array(m) # shape: (frame_count,)

    # divide video into num_frames segments, calculate the mean and median of each segment
    # retun the indices of the frames with that have mse closest to the mean and median of each segment
    segment_size = frame_count / num_frames
    segment_means = []
    segment_medians = []
    for i in range(num_frames):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size) if i != num_frames - 1 else frame_count
        segment = m[start_idx:end_idx]
        segment_means.append(np.mean(segment))
        segment_medians.append(np.median(segment))
    segment_means = np.array(segment_means)
    segment_medians = np.array(segment_medians)
    selected_med = []
    selected_mean = []
    for i in range(num_frames):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size) if i != num_frames - 1 else frame_count
        segment = m[start_idx:end_idx]
        mean_idx = start_idx + np.argmin(np.abs(segment - segment_means[i]))
        median_idx = start_idx + np.argmin(np.abs(segment - segment_medians[i]))
        # choose the one closer to the center of the segment
        center_idx = (start_idx + end_idx) // 2
        selected_med.append(median_idx)
        selected_mean.append(mean_idx)

    # Sorted all frames indices from m values, form low to high
    sorted_indices = np.argsort(m)
    # Divide sorted indices into num_frames segments, select the center index of each segment
    selected_uniform = []
    for i in range(num_frames):
        start_idx = int(i * len(sorted_indices) / num_frames)
        end_idx = int((i + 1) * len(sorted_indices) / num_frames) if i != num_frames - 1 else len(sorted_indices)
        segment = sorted_indices[start_idx:end_idx]
        center_idx = segment[len(segment) // 2]
        selected_uniform.append(center_idx)
    selected_uniform = np.sort(selected_uniform)

    # Selected top num_frames based on m
    selected_top = np.argsort(-m)[:num_frames]
    selected_top = np.sort(selected_top)  # sort to maintain temporal order
    return selected_uniform, selected_med, selected_mean, selected_top

    


def video_diff_sampler(video_path, num_frames=16):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    # print(f"Total frames in video {video_path}: {total_frames}")
    if total_frames < num_frames:
        # duplicate frames if video is shorter than num_frames 
        frame_indices = list(range(total_frames))
        while len(frame_indices) < num_frames:
            frame_indices += frame_indices
        frame_indices = frame_indices[:num_frames]
        frame_top = frame_indices
        frame_mean = frame_indices
        frame_med = frame_indices
        frame_uniform = frame_indices
    else:
        (frame_uniform, frame_med, frame_mean, frame_top) = frame_diff_sampling_video(video_path, max_frames=None, num_frames=num_frames)
    
    return frame_uniform, frame_med, frame_mean, frame_top

from os import path as osp
import glob
def frame_diff_sample_datasets(data_dir, dataset_info, dataset_infor_frame, num_frames=16):
    dataset_infor_mse_top = dataset_infor_frame + "top" + str(num_frames) + ".txt"
    dataset_infor_mse_uniform = dataset_infor_frame + "uniform" + str(num_frames) + ".txt"
    dataset_infor_mse_median = dataset_infor_frame + "median" + str(num_frames) + ".txt"
    dataset_infor_mse_mean = dataset_infor_frame + "mean" + str(num_frames) + ".txt"
    if isinstance(dataset_info, list):
            infos = dataset_info
    else:
        try:
            i = 0
            with open(dataset_info, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    fileid, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(data_dir, fileid)
                    # print("Processing video:", filename)
                    frame_uniform, frame_med, frame_mean, frame_top = video_diff_sampler(filename, num_frames=num_frames)
                    # i += 1
                    # if i < 1140:
                    #     continue
                    # print fileid, label, frame_indices to file dataset_infor_lgn
                    with open(dataset_infor_mse_top, "a") as fout16, open(dataset_infor_mse_uniform, "a") as fout8, open(dataset_infor_mse_median, "a") as fout4, open(dataset_infor_mse_mean, "a") as fout_mean:
                        fout16.write(f"{fileid},{label},{','.join(map(str, frame_top))}\n")
                        fout8.write(f"{fileid},{label},{','.join(map(str, frame_uniform))}\n")
                        fout4.write(f"{fileid},{label},{','.join(map(str, frame_med))}\n")
                        fout_mean.write(f"{fileid},{label},{','.join(map(str, frame_mean))}\n")

        except Exception as e:
            print(f"Error processing dataset_info: {e}")
    
    return 

import sys
from PIL import Image
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", "--dir", type=str, 
                        default="../../../VQA-datasets/InTheWild/Youtube-UGC/", help="video data directory")
    parser.add_argument("-data_info", "--info", type=str, 
                        default="./examplar_data_labels/YouTubeUGC/mylabels_filtered.txt", help="video data info file")
    parser.add_argument("-output_info", "--output", type=str, 
                        default="./examplar_data_labels/YouTubeUGC/mylabels_filtered_diff", help="output video data info file with diff frames sampling")
    parser.add_argument("-num_frames", "--num_frames", type=int, default=8, help="number of frames to sample")

    args = parser.parse_args()
    print(args)
    data_dir = args.dir
    data_info = args.info
    dataset_infor_lgn = args.output
    num_frames = args.num_frames
    frame_diff_sample_datasets(data_dir, data_info, dataset_infor_lgn, num_frames=num_frames)
    print("Frame diff sampling completed.")
    print("Done. ===================================================== ")

if __name__ == "__main__":



    # video_path = "../../../VQA-datasets/InTheWild/LSVQ/yfcc-batch1/113.mp4"
    # print("Processing video:", video_path)
    # frame_indices = video_LGN_sampler(video_path, num_frames=16, segment=True)
    # print("Selected frame indices:", frame_indices)
    

    main()

# Example usage:
# image = np.array(Image.open('frame.png').convert('L')) / 255.0
# Y_ori, Lap_ori = frame_LGN_features(image)
