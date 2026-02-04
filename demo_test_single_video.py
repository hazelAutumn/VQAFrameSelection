
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
def video_LGN_features_decord(video_path, max_frames=None, num_frames=16, segment=False):
    # with PCA    
    dn_filts = DN_filters()  # your DN filter function
    vr = VideoReader(video_path, ctx=cpu())
    all_features = []
    frame_count = len(vr) if max_frames is None else min(max_frames, len(vr))

    for i in range(frame_count):
        frame = vr[i].asnumpy().astype(np.float32) / 255.0
        gray = 0.2989 * frame[...,0] + 0.5870 * frame[...,1] + 0.1140 * frame[...,2]
        Y_ori, _ = frame_LGN_features(gray, dn_filts)
        all_features.append(Y_ori)
    # shape of all_features: (num_frames, 6, H, W)

    # PCA reduction can be applied here if needed
    n_components = num_frames  # example number of components
    pca = PCA(n_components=n_components)
    all_features_flat = [np.concatenate([band.flatten() for band in features]) for features in all_features]
    all_features_flat = np.nan_to_num(all_features_flat, nan=0.0)
    all_features_reduced = pca.fit_transform(all_features_flat) # shape: (num_frames, n_components)
    distance = (all_features_reduced[1:, :] - all_features_reduced[:-1, :]) # shape: (num_frames,)
    distance = np.linalg.norm(distance, axis=1)  # shape: (num_frames-1,)
    if segment:
        # Start indexs of segments
        segment_length = frame_count // num_frames
        segment_indices = [i * segment_length for i in range(num_frames)]
        # Get the index with the highest distance in each segment
        selected_indices = []
        for start in segment_indices:
            end = min(start + segment_length, frame_count)
            segment_distances = distance[start:end]
            max_index_in_segment = np.argmax(segment_distances)
            selected_indices.append(start + max_index_in_segment)
    else:
        # Select top num_frames based on distance
        selected_indices = np.argsort(-distance)[:num_frames]
        selected_indices = np.sort(selected_indices)  # sort to maintain temporal order

    return selected_indices


def video_LGN_sampler(vr, num_frames=16, segment=False):
    total_frames = len(vr)
    # print(f"Total frames in video {video_path}: {total_frames}")
    if total_frames < num_frames:
        # duplicate frames if video is shorter than num_frames 
        frame_indices = list(range(total_frames))
        while len(frame_indices) < num_frames:
            frame_indices += frame_indices
        frame_indices = frame_indices[:num_frames]
    else:
        frame_indices = video_LGN_features_decord(video_path, max_frames=250, num_frames=num_frames, segment=segment)+1
    return frame_indices

from os import path as osp
import glob
def LGN_sampling_dataset(data_dir, dataset_info, dataset_infor_lgn, num_frames=16, segment=False):
    if isinstance(dataset_info, list):
            infos = dataset_info
    else:
        try:
            #i = 0
            with open(dataset_info, "r") as fin:
                for line in fin:
                    # i += 1
                    # if i < 178: #176: #31: #17: # skip first 16 lines (header)
                    #     continue
                    line_split = line.strip().split(",")
                    fileid, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(data_dir, fileid)
                    frame_indices = video_LGN_sampler(filename, num_frames=num_frames, segment=segment)
                    # print fileid, label, frame_indices to file dataset_infor_lgn
                    with open(dataset_infor_lgn, "a") as fout:
                        fout.write(f"{fileid},{label},{','.join(map(str, frame_indices))}\n")

        except Exception as e:
            print(f"Error processing dataset_info: {e}")
    
    return 

import sys
from PIL import Image
import numpy as np
import argparse


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



def sample_uniform(vr, num_frames=16):
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int).tolist()
    return frame_indices

def sample_random(vr, num_frames=16):
    total_frames = len(vr)
    total_frames = len(vr)
    frame_indices = sorted(np.random.choice(range(total_frames), size=num_frames, replace=False).tolist())
    return frame_indices

def sample_random_segment(vr, num_frames=16):
    total_frames = len(vr)
    segment_length = total_frames // num_frames
    frame_indices = []
    for i in range(num_frames):
        start = i * segment_length
        end = min(start + segment_length, total_frames)
        frame_idx = np.random.randint(start, end)
        frame_indices.append(frame_idx)
    frame_indices = sorted(frame_indices)
    return frame_indices

def sample_MSE_mean(vr, num_frames=16): 
    frame_count = len(vr)
    diff_pre = vr[:frame_count-1].asnumpy().astype(np.float32) - vr[1:frame_count].asnumpy().astype(np.float32)
    # MSE per frame
    mse_per_frame= np.mean(diff_pre ** 2, axis=(1,2,3))  # shape: (frame_count-1,)
    m = [(mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame)-2)]
    # m = mse_per_frame[i] if i == 0 or i == len(mse_per_frame)-1 else (mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame))
    m = [mse_per_frame[0]] + m + [mse_per_frame[-1]]
    m = np.array(m) # shape: (frame_count,)

    # divide video into num_frames segments, calculate the mean of each segment
    # retun the indices of the frames with that have mse closest to the mean of each segment
    segment_size = frame_count / num_frames
    segment_means = []
    for i in range(num_frames):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size) if i != num_frames - 1 else frame_count
        segment = m[start_idx:end_idx]
        segment_means.append(np.mean(segment))
    segment_means = np.array(segment_means)
    selected_mean = []
    for i in range(num_frames):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size) if i != num_frames - 1 else frame_count
        segment = m[start_idx:end_idx]
        mean_idx = start_idx + np.argmin(np.abs(segment - segment_means[i]))
        selected_mean.append(mean_idx)
    frame_indices = selected_mean
    return frame_indices

def sample_MSE_median(vr, num_frames=16):
    frame_count = len(vr)
    diff_pre = vr[:frame_count-1].asnumpy().astype(np.float32) - vr[1:frame_count].asnumpy().astype(np.float32)
    # MSE per frame
    mse_per_frame= np.mean(diff_pre ** 2, axis=(1,2,3))  # shape: (frame_count-1,)
    m = [(mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame)-2)]
    # m = mse_per_frame[i] if i == 0 or i == len(mse_per_frame)-1 else (mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame))
    m = [mse_per_frame[0]] + m + [mse_per_frame[-1]]
    m = np.array(m) # shape: (frame_count,)

    # divide video into num_frames segments, calculate the median of each segment
    # retun the indices of the frames with that have mse closest to the median of each segment
    segment_size = frame_count // num_frames
    segment_medians = []
    for i in range(num_frames):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size) if i != num_frames - 1 else frame_count
        segment = m[start_idx:end_idx]
        segment_medians.append(np.median(segment))
    segment_medians = np.array(segment_medians)
    selected_median = []
    for i in range(num_frames):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size) if i != num_frames - 1 else frame_count
        segment = m[start_idx:end_idx]
        median_idx = start_idx + np.argmin(np.abs(segment - segment_medians[i]))
        selected_median.append(median_idx)
    frame_indices = selected_median
    return frame_indices

def sample_MSE_uniform(vr, num_frames=16):
    frame_count = len(vr)
    diff_pre = vr[:frame_count-1].asnumpy().astype(np.float32) - vr[1:frame_count].asnumpy().astype(np.float32)
    # MSE per frame
    mse_per_frame= np.mean(diff_pre ** 2, axis=(1,2,3))  # shape: (frame_count-1,)
    m = [(mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame)-2)]
    # m = mse_per_frame[i] if i == 0 or i == len(mse_per_frame)-1 else (mse_per_frame[i]+mse_per_frame[i+1])/2 for i in range(len(mse_per_frame))
    m = [mse_per_frame[0]] + m + [mse_per_frame[-1]]
    m = np.array(m) # shape: (frame_count,)

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
    #selected_uniform = np.sort(selected_uniform)
    frame_indices = selected_uniform
    return frame_indices

def sample_consecutive(vr, num_frames=16):
    total_frames = len(vr)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        start_idx = np.random.randint(0, total_frames - num_frames + 1)
        frame_indices = list(range(start_idx, start_idx + num_frames))
    return frame_indices


#video_path = "../../../VQA-datasets/InTheWild/Youtube-UGC/all-videos/Lecture_360P-2276_crf_10_ss_00_t_20.0.mp4"
#video_path = "../../../VQA-datasets/InTheWild/Youtube-UGC/all-videos/CoverSong_720P-10f1_crf_10_ss_00_t_20.0.mp4"
#video_path = "../../../VQA-datasets/InTheWild/Youtube-UGC/all-videos/HowTo_1080P-0267_crf_10_ss_00_t_20.0.mp4"
#video_path = "../../../VQA-datasets/InTheWild/Youtube-UGC/all-videos/Sports_2160P-0455_crf_10_ss_00_t_20.0.mp4"
video_path = "../../../VQA-datasets/InTheWild/LSVQ/yfcc-batch1/756.mp4"
video_path = "../../../VQA-datasets/Konvid-1k/KoNViD_1k_videos/4323977167.mp4"


#video_path = "../../../VQA-datasets/Konvid-1k/KoNViD_1k_videos/4542323058.mp4"
num_frames = 8
vr = VideoReader(video_path)


def code_to_run():
    frame_indices = video_LGN_sampler(vr, num_frames=num_frames)
    return frame_indices

import timeit
def main():



    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-10f1_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-2d05_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-4006_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-449f_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-5689_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-5ba2_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-60d3_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-6626_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-6b8c_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-6d6a_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-6d81_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/CoverSong_720P-7360_crf_10_ss_00_t_20.0.mp4"


    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-0267_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-03fd_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-13aa_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-1f3b_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-36a9_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-4d7b_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-52bb_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-55d1_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-63e4_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-63ec_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/HowTo_1080P-64f7_crf_10_ss_00_t_20.0.mp4"


    #video_path = "../../../VQA-datasets/InTheWild/Youtube-UGC/all-videos/Lecture_360P-2276_crf_10_ss_00_t_20.0.mp4"
    # video_path="../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-27db_crf_10_ss_00_t_20.0.mp4"
    # video_path="../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-2c55_crf_10_ss_00_t_20.0.mp4"
    # video_path="../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-2d1f_crf_10_ss_00_t_20.0.mp4"
    # video_path="../../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-30eb_crf_10_ss_00_t_20.0.mp4"
    # video_path2="../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-311d_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-3590_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-44c2_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-4bb4_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-4f00_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-506c_crf_10_ss_00_t_20.0.mp4"
    # video_path = "../../VQA-datasets/InTheWild/YouTube-UGC/all-videos/Lecture_360P-5594_crf_10_ss_00_t_20.0.mp4"


    
    print("video_path:", video_path, " total frames:", len(vr), "resolution", resolution:=vr[0].shape)

    code_to_run2 = "code_to_run()"
    total_time = timeit.timeit(code_to_run2, globals=globals(), number=100)
    print(f"Average time for sample_MSE_uniform over 100 runs: {total_time/100:.5f} seconds")

if __name__ == "__main__":


    main()
