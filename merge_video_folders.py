import os
import shutil
import argparse

def merge_videos(source_dir, target_dir):
    for file in os.listdir(source_dir):
        if file.endswith('.mp4'):  
            source_file = os.path.join(source_dir, file)
            target_file = os.path.join(target_dir, file)

            if os.path.exists(target_file):
                base, extension = os.path.splitext(file)
                count = 1
                new_file = f"{base}_{count}{extension}"
                target_file = os.path.join(target_dir, new_file)

                while os.path.exists(target_file):
                    count += 1
                    new_file = f"{base}_{count}{extension}"
                    target_file = os.path.join(target_dir, new_file)

            shutil.move(source_file, target_file)
            print(f"Moved: {source_file} to {target_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge MP4 files from two directories into a single target directory.")
    parser.add_argument("--source_dir1", type=str, help="Path to the first source video directory")
    parser.add_argument("--source_dir2", type=str, help="Path to the second source video directory")
    parser.add_argument("--target_dir", type=str, help="Path to the target video directory")

    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    merge_videos(args.source_dir1, args.target_dir)
    merge_videos(args.source_dir2, args.target_dir)

    print("Merging complete.")

if __name__ == "__main__":
    main()