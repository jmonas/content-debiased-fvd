from cdfvd import fvd
evaluator = fvd.cdfvd('i3d', ckpt_path=None, device="cpu")
gen_path = "../Vista/logs_256/2024-08-15T13-09-45_training-vista_phase1_fps5_frames25_dataset1xgpt_competition/images/train/samples_mp4"
gt_path = "../Vista/logs_256/2024-08-15T13-09-45_training-vista_phase1_fps5_frames25_dataset1xgpt_competition/images/train/inputs_mp4"
evaluator.compute_real_stats(evaluator.load_videos(gt_path, resolution=256, sequence_length=25, data_type='video_folder', conditioning_frames=6, subset_num = 50))
print("real stats computed")
evaluator.compute_fake_stats(evaluator.load_videos(gen_path, resolution=256, sequence_length=25, data_type='video_folder', conditioning_frames=6, subset_num = 50))
print("gen stats computed")
score = evaluator.compute_fvd_from_stats()

print(f"id3: {score}")

# evaluator = fvd.cdfvd('videomae', ckpt_path=None)
# evaluator.compute_real_stats(evaluator.load_videos('path/to/realvideos/'))
# evaluator.compute_fake_stats(evaluator.load_videos('path/to/fakevideos/'))
# score = evaluator.compute_fvd_from_stats()

# print(f"videomae: {score}")
