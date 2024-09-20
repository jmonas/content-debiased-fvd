from calc_fvd import compute_fvd
from cdfvd import fvd
import json
import argparse
import os
import random
import gc
import argparse
import imageio.v3 as iio
import numpy as np
from einops import rearrange
from torchvision.transforms import v2
from tqdm.auto import tqdm
import os
import torchmetrics.image
from calculate_lpips import calculate_lpips

import torch
from calculate_fvd import calculate_fvd
from pathlib import Path



def mp4_to_torch(mp4_path):
    frames_np = iio.imread(mp4_path, plugin="pyav")
    frames_torch = rearrange(torch.from_numpy(frames_np), "b h w c -> b c h w")
    return v2.functional.to_dtype(frames_torch, scale=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FVD for generated and ground truth videos.")
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing generations for the model."
             "There should be subdirectories `samples_mp4` (generations) and `targets_mp4` (ground truth)."
    )
    parser.add_argument(
        "--num_cond_frames",
        type=int,
        required=True,
        help="Number of conditioning frames, will be excluded from calculations."
    )
    # parser.add_argument(
    #     "--dataset_size",
    #     type=int,
    # )
    # parser.add_argument("--gen_path", type=str, help="Path to the generated videos directory")
    # parser.add_argument("--gt_path", type=str, help="Path to the ground truth videos directory")
    # parser.add_argument("--resolution", type=int, default=256, help="Resolution of the videos (default: 256)")
    # parser.add_argument("--sequence_length", type=int, default=25, help="Number of frames in each video sequence (default: 25)")
    # parser.add_argument("--data_type", type=str, default='video_folder', help="Type of the data input (default: 'video_folder')")
    # parser.add_argument("--conditioning_frames", type=int, default=6, help="Number of conditioning frames (default: 3)")
    parser.add_argument("--output", type=str, default="./fvd_results")
    parser.add_argument("--max_files", type=int, default=None)
    # parser.add_argument("--subset_num", type=int, default=50, help="Number of samples to compute FVD on (default: 50)")

    args = parser.parse_args()
    



    img_dir = Path(args.img_dir)
    samples_dir = img_dir / "virtual" / "videos"
    targets_dir = img_dir / "real" / "videos"

    all_samples = os.listdir(samples_dir)
    all_targets = os.listdir(targets_dir)
    

    samples_set = set(os.path.basename(file) for file in all_samples)
    targets_set = set(os.path.basename(file) for file in all_targets)


    common_files = samples_set.intersection(targets_set)
    common_files = list(common_files)

    print("len common files: ", len(common_files))

    if args.max_files:
        common_files[:args.max_files] = common_files
        print("remaining files", len(common_files))
    random.seed(0) 

    final_samples = [os.path.join(samples_dir, sample_name) for sample_name in common_files]
    final_targets = [os.path.join(targets_dir, sample_name) for sample_name in common_files]

    videos1 = torch.stack([mp4_to_torch(sample)[args.num_cond_frames:] for sample in final_samples])
    videos2 = torch.stack([mp4_to_torch(target)[args.num_cond_frames:] for target in final_targets])
    print("start lpips")
    lpips= calculate_lpips(videos1, videos2, "cuda")
    print(f"LPIPS: {np.mean(list(lpips['value'].values()))}")
    fvd_score = calculate_fvd(videos1, videos2, "cuda", method='styleganv')
    print(f" FVD: {fvd_score} || LPIPS : {np.mean(list(lpips['value'].values()))}")

