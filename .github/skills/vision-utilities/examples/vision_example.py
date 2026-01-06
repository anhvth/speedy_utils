import os

import numpy as np
from PIL import Image

from vision_utils.io_utils import ImageMmap, read_images


def create_dummy_images(n=5):
    paths = []
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        path = f"dummy_{i}.jpg"
        img.save(path)
        paths.append(path)
    return paths

def main():
    # 1. Create dummy images
    paths = create_dummy_images()
    print(f"Created {len(paths)} dummy images.")

    try:
        # 2. Read images (CPU fallback likely)
        print("Reading images...")
        images = read_images(paths, hw=(50, 50))
        print(f"Loaded {len(images)} images.")
        print(f"Shape of first image: {images[paths[0]].shape}")

        # 3. Create Mmap Dataset
        print("Creating Mmap Dataset...")
        dataset = ImageMmap(paths, size=(64, 64))
        print(f"Dataset size: {len(dataset)}")
        print(f"Item 0 shape: {dataset[0].shape}")

    finally:
        # Cleanup
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
        # Note: Mmap cache files in .cache/ are left behind

if __name__ == "__main__":
    main()
