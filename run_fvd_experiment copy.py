from calc_fvd import compute_fvd
from cdfvd import fvd
import json
import argparse
import os
import random
import gc
import argparse
import imageio.v3 as iio

from einops import rearrange
from torchvision.transforms import v2
from tqdm.auto import tqdm
import os
import torchmetrics.image

import torch
from calculate_fvd import calculate_fvd
from pathlib import Path


def powers_of_two(max_value):
    n = 8
    while (2 ** n) <= max_value:
        yield 2 ** n
        n += 1

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
    # parser.add_argument("--gen_path", type=str, help="Path to the generated videos directory")
    # parser.add_argument("--gt_path", type=str, help="Path to the ground truth videos directory")
    # parser.add_argument("--resolution", type=int, default=256, help="Resolution of the videos (default: 256)")
    # parser.add_argument("--sequence_length", type=int, default=25, help="Number of frames in each video sequence (default: 25)")
    # parser.add_argument("--data_type", type=str, default='video_folder', help="Type of the data input (default: 'video_folder')")
    # parser.add_argument("--conditioning_frames", type=int, default=6, help="Number of conditioning frames (default: 3)")
    # parser.add_argument("--output", type=str, default="./fvd_results", help="Number of conditioning frames (default: 3)")

    # parser.add_argument("--subset_num", type=int, default=50, help="Number of samples to compute FVD on (default: 50)")

    args = parser.parse_args()
    



    img_dir = Path(args.img_dir)
    samples_dir = img_dir / "virtual" / "videos"
    targets_dir = img_dir / "real" / "videos"

    all_samples = os.listdir(samples_dir)


    fvd_results ={}
    num_vids = len(os.listdir(args.gen_path))
    num_vids_gt = len(os.listdir(args.gt_path))
    assert num_vids == num_vids_gt
    for subset_num in powers_of_two(num_vids):
        results = []
        for run_idx in range(10):
            random.seed(0) 
            print(f"Runnning FVD for {subset_num} videos (experiment #{run_idx}):", subset_num)


            final_samples = random.sample(all_samples, subset_num)
            final_targets = [sample.replace("virtual", "real") for sample in final_samples]
            assert all((targets_dir / final_target).exists() for final_target in final_targets), "Missing ground truth."
            videos1 = torch.stack([mp4_to_torch(samples_dir / sample)[args.num_cond_frames:] for sample in final_samples])
            videos2 = torch.stack([mp4_to_torch(targets_dir / target)[args.num_cond_frames:] for target in final_targets])


            fvd_score = calculate_fvd(videos1, videos2, "cuda", method='styleganv')

            print(fvd_score)
            # fvd_score = compute_fvd(args.gen_path, args.gt_path, resolution=args.resolution, sequence_length=args.sequence_length, data_type=args.data_type, conditioning_frames=args.conditioning_frames, subset_num = random_indexes)
            # results.append(result['fvd'] )
        fvd_results[subset_num] = results

    os.makedirs(args.output, exist_ok=True)
    print(fvd_results)
    with open(args.output, 'w') as json_file:
        json.dump(fvd_results, json_file, indent=4)
