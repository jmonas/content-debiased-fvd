from cdfvd import fvd
evaluator = fvd.cdfvd('i3d', ckpt_path=None, device="cpu")

# gen_path = "../Vista/logs_256/2024-08-15T13-09-45_training-vista_phase1_fps5_frames25_dataset1xgpt_competition/images/train/samples_mp4"
# gt_path = "../Vista/logs_256/2024-08-15T13-09-45_training-vista_phase1_fps5_frames25_dataset1xgpt_competition/images/train/inputs_mp4"
gen_path = "../repos/vista/logs/2024-08-26T06-34-54_training-vista_ft_actions/images/train/samples_mp4"
gt_path = "../repos/vista/logs/2024-08-26T06-34-54_training-vista_ft_actions/images/train/inputs_mp4"
# gen_path = "../repos/vista/logs/2024-08-24T05-35-59_training-vista_ft_actions/images/train/samples_mp4"
# gt_path = "../repos/vista/logs/2024-08-24T05-35-59_training-vista_ft_actions/images/train/inputs_mp4"
evaluator.compute_real_stats(evaluator.load_videos(gt_path, resolution=256, sequence_length=25, data_type='video_folder', conditioning_frames=3, subset_num = 50))
print("real stats computed")
evaluator.compute_fake_stats(evaluator.load_videos(gen_path, resolution=256, sequence_length=25, data_type='video_folder', conditioning_frames=3, subset_num = 50))
print("gen stats computed")
score = evaluator.compute_fvd_from_stats()

print(f"id3: {score}")

# evaluator = fvd.cdfvd('videomae', ckpt_path=None)
# evaluator.compute_real_stats(evaluator.load_videos(gt_path, resolution=256, sequence_length=25, data_type='video_folder', conditioning_frames=6, subset_num = 50))
# evaluator.compute_fake_stats(evaluator.load_videos(gen_path, resolution=256, sequence_length=25, data_type='video_folder', conditioning_frames=6, subset_num = 50))
# score = evaluator.compute_fvd_from_stats()

# print(f"videomae: {score}")
