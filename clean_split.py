import os
import shutil
import random

def create_clean_split(original_folder, output_folder, split_ratio=0.8):
    classes = ['infected', 'notinfected']

    for cls in classes:
        files = os.listdir(os.path.join(original_folder, cls))
        files = list(set(files))  # remove any duplicates by name (as extra safety)
        random.shuffle(files)

        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        test_files = files[split_index:]

        for mode, dataset in [('train', train_files), ('test', test_files)]:
            out_dir = os.path.join(output_folder, mode, cls)
            os.makedirs(out_dir, exist_ok=True)
            for f in dataset:
                src = os.path.join(original_folder, cls, f)
                dst = os.path.join(out_dir, f)
                shutil.copyfile(src, dst)

    print("✅ Clean dataset split created successfully!")

# ORIGINAL unstructured folder (where all images are together under infected/notinfected)
original_data_path = "data/train"  # contains ALL data
output_data_path = "clean_data"    # new folder that will have clean split

create_clean_split(original_data_path, output_data_path)
