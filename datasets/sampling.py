
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


def video_LGN_sampler(video_path, num_frames=16, segment=False):
    vr = VideoReader(video_path)
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", "--dir", type=str, 
                        default="../../../VQA-datasets/InTheWild/LSVQ/", help="video data directory")
    parser.add_argument("-data_info", "--info", type=str, 
                        default="./examplar_data_labels/LSVQ/labels_mytrain.txt", help="video data info file")
    parser.add_argument("-output_info", "--output", type=str, 
                        default="./examplar_data_labels/LSVQ/labels_mytrain_lgn16.txt", help="output video data info file with LGN sampling")
    parser.add_argument("-num_frames", "--num_frames", type=int, default=16, help="number of frames to sample")
    parser.add_argument("-segment", "--segment", type=bool, default=False, help="use segment sampling")

    args = parser.parse_args()
    data_dir = args.dir
    data_info = args.info
    dataset_infor_lgn = args.output
    num_frames = args.num_frames
    segment = args.segment
    LGN_sampling_dataset(data_dir, data_info, dataset_infor_lgn, num_frames=num_frames, segment=segment)
    print("LGN sampling completed.")
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
