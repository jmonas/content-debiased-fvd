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


def powers_of_two(n, start = 32):
    """Generate powers of two up to and including n."""
    k = start
    while k <= n:
        yield k
        k *= 2
    if k // 2 < n:
        yield n

def mp4_to_torch(mp4_path):
    frames_np = iio.imread(mp4_path, plugin="pyav")
    frames_torch = rearrange(torch.from_numpy(frames_np), "b h w c -> b c h w")
    return v2.functional.to_dtype(frames_torch, scale=True)

def chunk_list(data, size):
    """Yield successive size-sized chunks from data."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

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
    parser.add_argument(
        "--dataset_size",
        type=int,
    )
    # parser.add_argument("--gen_path", type=str, help="Path to the generated videos directory")
    # parser.add_argument("--gt_path", type=str, help="Path to the ground truth videos directory")
    # parser.add_argument("--resolution", type=int, default=256, help="Resolution of the videos (default: 256)")
    # parser.add_argument("--sequence_length", type=int, default=25, help="Number of frames in each video sequence (default: 25)")
    # parser.add_argument("--data_type", type=str, default='video_folder', help="Type of the data input (default: 'video_folder')")
    # parser.add_argument("--conditioning_frames", type=int, default=6, help="Number of conditioning frames (default: 3)")
    parser.add_argument("--output", type=str, default="./fvd_results", help="Number of conditioning frames (default: 3)")

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
    common_files_list = list(common_files)
    # if args.dataset_size:
    #     groups = list(chunk_list(common_files_list, args.dataset_size))[:-1]

    print("len common files: ", len(common_files))



    fvd_results ={"sample_size": len(common_files),"fvd":{}, "lpips":{}}
    random.seed(0) 

    print("sampling", [32, 64, 128, 256])
            

    for subset_num in [250, 500, 1000, 1500]:
        groups = list(chunk_list(common_files_list, subset_num))[:-1]
        group_res_fvd = []
        group_res_lpips = []

        for group in groups:
            results_fvd = []
            results_lpips = []
            for run_idx in range(1):
                selected_files = random.sample(common_files, subset_num)

                final_samples = [os.path.join(samples_dir, sample_name) for sample_name in selected_files]
                final_targets = [os.path.join(targets_dir, sample_name) for sample_name in selected_files]

                videos1 = torch.stack([mp4_to_torch(sample)[args.num_cond_frames:] for sample in final_samples])
                videos2 = torch.stack([mp4_to_torch(target)[args.num_cond_frames:] for target in final_targets])
                lpips= calculate_lpips(videos1, videos2, "cuda")


                # fvd_score = calculate_fvd(videos1, videos2, "cuda", method='styleganv')

                print(f"{subset_num} videos (experiment #{run_idx}) LPIPS : {np.mean(list(lpips['value'].values()))}")
                # results_fvd.append(fvd_score)
                results_lpips.append(np.mean(list(lpips["value"].values())))

                # fvd_score = compute_fvd(args.gen_path, args.gt_path, resolution=args.resolution, sequence_length=args.sequence_length, data_type=args.data_type, conditioning_frames=args.conditioning_frames, subset_num = random_indexes)
                # results.append(result['fvd'] )
            # fvd_results["fvd"][subset_num][group] = results_fvd
            # fvd_results["lpips"][subset_num][group] = results_lpips
            group_res_fvd.append(results_fvd)
            group_res_lpips.append(results_lpips)

        print("-----------------------------")
        print(subset_num)
        print("FVD")
        fvd_var = []
        lpips_var =[]
        for data in group_res_fvd:
            data_array = np.array(data)
            average = np.mean(data_array)

            # Calculate the variance
            variance = np.var(data_array, ddof=1)  # ddof=1 provides the sample variance
            fvd_var.append(variance)

            print(f"{data}----Average: {average}----Variance: {variance}")
        print("LPIPS")
        for data in group_res_lpips:
            data_array = np.array(data)
            average = np.mean(data_array)

            # Calculate the variance
            variance = np.var(data_array, ddof=1)  # ddof=1 provides the sample variance
            lpips_var.append(variance)


            print(f"{data}----Average: {average}----Variance: {variance}")

        print()
    

    os.makedirs(args.output, exist_ok=True)
    print(fvd_results)
    with open(args.output, 'w') as json_file:
        json.dump({"fvd_var": fvd_var, "lpips_var":lpips_var}, json_file, indent=4)
