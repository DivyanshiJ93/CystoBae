import os
import hashlib

def get_image_hashes(folder):
    hashes = set()
    for root, _, files in os.walk(folder):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'rb') as img:
                    hash_val = hashlib.md5(img.read()).hexdigest()
                    hashes.add(hash_val)
            except:
                pass
    return hashes

train_hashes = get_image_hashes("clean_data/train")
test_hashes = get_image_hashes("clean_data/test")

common = train_hashes.intersection(test_hashes)
print(f"🔍 Found {len(common)} duplicate images between train and test sets.")
