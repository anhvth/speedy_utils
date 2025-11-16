from __future__ import annotations

# type: ignore
import os
import time
from pathlib import Path
from typing import Sequence, Tuple, TYPE_CHECKING
from multiprocessing import cpu_count

import numpy as np
from PIL import Image
from speedy_utils import identify

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object


if TYPE_CHECKING:
    from nvidia.dali import fn, pipeline_def
    from nvidia.dali import types as dali_types
    from tqdm import tqdm


PathLike = str | os.PathLike


def _to_str_paths(paths: Sequence[PathLike]) -> list[str]:
    return [os.fspath(p) for p in paths]


def _validate_image(path: PathLike) -> bool:
    """
    Validate if an image file is readable and not corrupted.
    Returns True if valid, False otherwise.
    """
    from PIL import Image

    path = os.fspath(path)

    if not os.path.exists(path):
        return False

    try:
        with Image.open(path) as img:
            img.verify()  # Verify it's a valid image
        # Re-open after verify (verify closes the file)
        with Image.open(path) as img:
            img.load()  # Actually decode the image data
        return True
    except Exception:
        return False


def read_images_cpu(
    paths: Sequence[PathLike],
    hw: tuple[int, int] | None = None,
) -> dict[str, 'np.ndarray | None']:
    """
    CPU image loader using Pillow.

    Returns dict mapping paths -> numpy arrays (H, W, C, RGB) or None for invalid images.

    Args:
        paths: Sequence of image file paths.
        hw: Optional (height, width) for resizing.
    """
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    str_paths = _to_str_paths(paths)

    # Pillow < 9.1.0 exposes resampling filters directly on Image
    resample_attr = getattr(Image, 'Resampling', Image)
    resample = resample_attr.BILINEAR

    target_size = None  # Pillow expects (width, height)
    if hw is not None:
        h, w = hw
        target_size = (w, h)

    result: dict[str, 'np.ndarray | None'] = {}
    for path in tqdm(str_paths, desc='Loading images (CPU)', unit='img'):
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                if target_size is not None:
                    img = img.resize(target_size, resample=resample)
                result[path] = np.asarray(img)
        except Exception as e:
            print(f'Warning: Failed to load {path}: {e}')
            result[path] = None
    return result


def read_images_gpu(
    paths: Sequence[PathLike],
    batch_size: int = 32,
    num_threads: int = 4,
    hw: tuple[int, int] | None = None,
    validate: bool = False,
    device: str = 'mixed',
    device_id: int = 0,
) -> dict[str, 'np.ndarray | None']:
    """
    GPU-accelerated image reader using NVIDIA DALI.

    Returns dict mapping paths -> numpy arrays (H, W, C, RGB) or None for invalid images.

    Args:
        paths: Sequence of image file paths.
        batch_size: Batch size for DALI processing.
        num_threads: Number of threads for DALI decoding.
        hw: Optional (height, width) for resizing.
        validate: If True, pre-validate images (slower).
        device: DALI decoder device: "mixed" (default), "cpu", or "gpu".
        device_id: GPU device id.
    """
    import numpy as np
    from nvidia.dali import fn, pipeline_def
    from nvidia.dali import types as dali_types

    str_paths = _to_str_paths(paths)

    if not str_paths:
        return {}

    result: dict[str, 'np.ndarray | None'] = {}
    valid_paths: list[str] = str_paths

    # Optional validation (slow but safer)
    if validate:
        from tqdm import tqdm

        print('Validating images...')
        tmp_valid: list[str] = []
        invalid_paths: list[str] = []

        for path in tqdm(str_paths, desc='Validating', unit='img'):
            if _validate_image(path):
                tmp_valid.append(path)
            else:
                invalid_paths.append(path)
                print(f'Warning: Skipping invalid/corrupted image: {path}')

        valid_paths = tmp_valid
        # pre-fill invalid paths with None
        for p in invalid_paths:
            result[p] = None

        if not valid_paths:
            print('No valid images found.')
            return result

    resize_h, resize_w = (None, None)
    if hw is not None:
        resize_h, resize_w = hw  # (H, W)

    files_for_reader = list(valid_paths)

    @pipeline_def
    def pipe():
        # Keep deterministic order to match valid_paths
        jpegs, _ = fn.readers.file(
            files=files_for_reader,
            random_shuffle=False,
            name='Reader',
        )
        imgs = fn.decoders.image(jpegs, device=device, output_type=dali_types.RGB)
        if resize_h is not None and resize_w is not None:
            # DALI resize expects (resize_x=width, resize_y=height)
            imgs_resized = fn.resize(
                imgs,
                resize_x=resize_w,
                resize_y=resize_h,
                interp_type=dali_types.INTERP_TRIANGULAR,
            )
            return imgs_resized
        return imgs

    dali_pipe = pipe(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=2,
    )
    dali_pipe.build()

    imgs: list['np.ndarray'] = []
    num_files = len(valid_paths)
    num_batches = (num_files + batch_size - 1) // batch_size

    from tqdm import tqdm

    for _ in tqdm(range(num_batches), desc='Decoding (DALI)', unit='batch'):
        (out,) = dali_pipe.run()
        out = out.as_cpu()
        for i in range(len(out)):
            imgs.append(np.array(out.at(i)))

    # Handle possible padding / extra samples
    if len(imgs) < num_files:
        print(
            f'Warning: DALI returned fewer samples ({len(imgs)}) than expected ({num_files}).'
        )
    if len(imgs) > num_files:
        imgs = imgs[:num_files]

    # Map valid images to result
    for path, img in zip(valid_paths, imgs, strict=False):
        result[path] = img

    return result


def read_images(
    paths: Sequence[PathLike],
    batch_size: int = 32,
    num_threads: int = 4,
    hw: tuple[int, int] | None = None,
    validate: bool = False,
    device: str = 'mixed',
    device_id: int = 0,
) -> dict[str, 'np.ndarray | None']:
    """
    Fast image reader that tries GPU (DALI) first, falls back to CPU (Pillow).

    Returns dict mapping paths -> numpy arrays (H, W, C, RGB) or None for invalid images.

    Args:
        paths: Sequence of image file paths.
        batch_size: Batch size for DALI processing (GPU only).
        num_threads: Number of threads for decoding (GPU only).
        hw: Optional (height, width) for resizing.
        validate: If True, pre-validate images before GPU processing (slower).
        device: DALI decoder device: "mixed", "cpu", or "gpu".
        device_id: GPU device id for DALI.
    """
    str_paths = _to_str_paths(paths)

    if not str_paths:
        return {}

    try:
        return read_images_gpu(
            str_paths,
            batch_size=batch_size,
            num_threads=num_threads,
            hw=hw,
            validate=validate,
            device=device,
            device_id=device_id,
        )
    except Exception as exc:
        print(f'GPU loading failed ({exc}), falling back to CPU...')
        return read_images_cpu(str_paths, hw=hw)





class ImageMmap(Dataset):
    """
    One-time build + read-only mmap dataset.

    - First run (no mmap file): read all img_paths -> resize -> write mmap.
    - Next runs: only read from mmap (no filesystem image reads).
    """

    def __init__(
        self,
        img_paths: Sequence[str | os.PathLike],
        size: Tuple[int, int] = (224, 224),
        mmap_path: str | os.PathLike | None = None,
        dtype: np.dtype = np.uint8,
        C=3,
        safe: bool = True,
    ) -> None:
        self.imgpath2idx = {str(p): i for i, p in enumerate(img_paths)}
        self.img_paths = [str(p) for p in img_paths]
        self.H, self.W = size
        self.C = C
        self.n = len(self.img_paths)
        self.dtype = np.dtype(dtype)
        self.safe = safe

        # Generate default mmap path if not provided
        if mmap_path is None:
            hash_idx = identify(''.join(self.img_paths))
            mmap_path = Path('.cache') / f'mmap_dataset_{hash_idx}.dat'
        
        self.mmap_path = Path(mmap_path)
        self.hash_path = Path(str(self.mmap_path) + '.hash')
        self.lock_path = Path(str(self.mmap_path) + '.lock')
        self.shape = (self.n, self.H, self.W, self.C)

        if self.n == 0:
            raise ValueError("Cannot create ImageMmap with empty img_paths list")

        # Calculate hash of image paths
        current_hash = identify(self.img_paths)
        needs_rebuild = False
        
        if not self.mmap_path.exists():
            needs_rebuild = True
            print("Mmap file does not exist, building cache...")
        elif not self.hash_path.exists():
            needs_rebuild = True
            print("Hash file does not exist, rebuilding cache...")
        else:
            # Check if hash matches
            stored_hash = self.hash_path.read_text().strip()
            if stored_hash != current_hash:
                needs_rebuild = True
                print(f"Hash mismatch (stored: {stored_hash[:16]}..., current: {current_hash[:16]}...), rebuilding cache...")
        
        # Verify file size matches expected
        expected_bytes = np.prod(self.shape) * self.dtype.itemsize
        if self.mmap_path.exists():
            actual_size = self.mmap_path.stat().st_size
            if actual_size != expected_bytes:
                needs_rebuild = True
                print(f"Mmap file size mismatch (expected: {expected_bytes}, got: {actual_size}), rebuilding cache...")
        
        if needs_rebuild:
            self._build_cache_with_lock(current_hash)

        # runtime: always open read-only; assume cache is complete
        self.data = np.memmap(
            self.mmap_path,
            dtype=self.dtype,
            mode="r",
            shape=self.shape,
        )

    # --------------------------------------------------------------------- #
    # Build phase (only on first run)
    # --------------------------------------------------------------------- #
    def _build_cache_with_lock(self, current_hash: str, num_workers: int = None) -> None:
        """Build cache with lock file to prevent concurrent disk writes"""
        import fcntl
        
        self.mmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to acquire lock file
        lock_fd = None
        try:
            lock_fd = open(self.lock_path, 'w')
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # We got the lock, build the cache
            self._build_cache(current_hash, num_workers)
            
        except BlockingIOError:
            # Another process is building, wait for it
            print("Another process is building the cache, waiting...")
            if lock_fd:
                lock_fd.close()
            lock_fd = open(self.lock_path, 'w')
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)  # Wait for lock
            print("Cache built by another process!")
            
        finally:
            if lock_fd:
                lock_fd.close()
            if self.lock_path.exists():
                try:
                    self.lock_path.unlink()
                except:
                    pass
    
    def _build_cache(self, current_hash: str, num_workers: int = None) -> None:
        from tqdm import tqdm
        
        # Pre-allocate the file with the required size
        total_bytes = np.prod(self.shape) * self.dtype.itemsize
        print(f"Pre-allocating {total_bytes / (1024**3):.2f} GB for mmap file...")
        with open(self.mmap_path, 'wb') as f:
            f.seek(total_bytes - 1)
            f.write(b'\0')
        
        mm = np.memmap(
            self.mmap_path,
            dtype=self.dtype,
            mode='r+',
            shape=self.shape,
        )
        
        # Process images in batches to avoid memory explosion
        batch_size = 4096
        num_batches = (self.n + batch_size - 1) // batch_size
        
        print(f"Loading {self.n} images in {num_batches} batches of up to {batch_size} images...")
        
        with tqdm(total=self.n, desc='Processing images', unit='img') as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.n)
                batch_paths = self.img_paths[start_idx:end_idx]
                
                # Load one batch at a time
                images_dict = read_images(
                    batch_paths,
                    hw=(self.H, self.W),
                    batch_size=32,
                    num_threads=num_workers or max(1, cpu_count() - 1),
                )
                
                # Write batch to mmap
                for local_idx, path in enumerate(batch_paths):
                    global_idx = start_idx + local_idx
                    img = images_dict.get(path)
                    
                    if img is None:
                        if self.safe:
                            raise ValueError(f'Failed to load image: {path}')
                        else:
                            # Failed to load, write zeros
                            print(f'Warning: Failed to load {path}, using zeros')
                            mm[global_idx] = np.zeros(
                                (self.H, self.W, self.C),
                                dtype=self.dtype
                            )
                    else:
                        # Ensure correct dtype
                        if img.dtype != self.dtype:
                            img = img.astype(self.dtype)
                        mm[global_idx] = img
                    
                    pbar.update(1)
                
                # Flush after each batch and clear memory
                mm.flush()
                del images_dict
        
        mm.flush()
        del mm  # ensure descriptor is closed
        
        # Save hash file
        self.hash_path.write_text(current_hash)
        print(f"Mmap cache built successfully! Hash saved to {self.hash_path}")

    def _load_and_resize(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.W, self.H), Image.BILINEAR)
        return np.asarray(img, dtype=self.dtype)

    # --------------------------------------------------------------------- #
    # Dataset API
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> np.ndarray:
        # At runtime: this is just a mmap read
        return np.array(self.data[idx])  # copy to normal ndarray
    
    def imread(self, image_path: str | os.PathLike) -> np.ndarray:
        idx = self.imgpath2idx.get(str(image_path))
        if idx is None:
            raise ValueError(f"Image path {image_path} not found in dataset")
        img =  np.array(self.data[idx])  # copy to normal ndarray
        summary = img.sum()
        assert summary > 0, f"Image at {image_path} appears to be all zeros"
        return img


__all__ = ['read_images', 'read_images_cpu', 'read_images_gpu', 'ImageMmap']
