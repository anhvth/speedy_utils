# type: ignore
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np
    from nvidia.dali import fn, pipeline_def
    from nvidia.dali import types as dali_types
    from PIL import Image
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


__all__ = ['read_images', 'read_images_cpu', 'read_images_gpu']
