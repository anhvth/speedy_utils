import os
import sys
import shutil
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_hf_cache_home():
    """Locate the Hugging Face cache directory."""
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"

def resolve_model_path(model_id, cache_dir):
    """Find the physical snapshot directory for the given model ID."""
    dir_name = "models--" + model_id.replace("/", "--")
    model_root = cache_dir / dir_name
    if not model_root.exists():
        raise FileNotFoundError(f"Model folder not found at: {model_root}")

    # 1. Try to find hash via refs/main
    ref_path = model_root / "refs" / "main"
    if ref_path.exists():
        with open(ref_path, "r") as f:
            commit_hash = f.read().strip()
        snapshot_path = model_root / "snapshots" / commit_hash
        if snapshot_path.exists():
            return snapshot_path
    
    # 2. Fallback to the newest snapshot folder
    snapshots_dir = model_root / "snapshots"
    if snapshots_dir.exists():
        subdirs = [x for x in snapshots_dir.iterdir() if x.is_dir()]
        if subdirs:
            return sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            
    raise FileNotFoundError(f"No valid snapshot found in {model_root}")

def copy_worker(src, dst):
    """Copy a single file, following symlinks to capture actual data."""
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # copy2 follows symlinks by default
        shutil.copy2(src, dst)
        return os.path.getsize(dst)
    except Exception as e:
        return str(e)

def cache_to_ram(model_id, shm_base, workers=64):
    """Parallel copy from HF cache to the specified RAM directory."""
    cache_home = get_hf_cache_home()
    src_path = resolve_model_path(model_id, cache_home)
    
    safe_name = model_id.replace("/", "_")
    dst_path = Path(shm_base) / safe_name

    # Check available space in shm
    shm_stats = shutil.disk_usage(shm_base)
    print(f"📦 Source: {src_path}", file=sys.stderr)
    print(f"🚀 Target RAM: {dst_path} (Available: {shm_stats.free/(1024**3):.1f} GB)", file=sys.stderr)

    files_to_copy = []
    for root, _, files in os.walk(src_path):
        for file in files:
            full_src = Path(root) / file
            rel_path = full_src.relative_to(src_path)
            files_to_copy.append((full_src, dst_path / rel_path))

    total_bytes = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(copy_worker, s, d): s for s, d in files_to_copy}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if isinstance(res, int):
                total_bytes += res
            if i % 100 == 0 or i == len(files_to_copy) - 1:
                print(f"   Progress: {i+1}/{len(files_to_copy)} files...", end="\r", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\n✅ Copied {total_bytes/(1024**3):.2f} GB in {elapsed:.2f}s", file=sys.stderr)
    return dst_path

def main():
    parser = argparse.ArgumentParser(description="vLLM RAM-cached loader", add_help=False)
    parser.add_argument("--model", type=str, required=True, help="HuggingFace Model ID")
    parser.add_argument("--shm-dir", type=str, default="/dev/shm", help="RAM disk mount point")
    parser.add_argument("--cache-workers", type=int, default=64, help="Threads for copying")
    parser.add_argument("--keep-cache", action="store_true", help="Do not delete files from RAM on exit")
    
    # Capture wrapper args vs vLLM args
    args, vllm_args = parser.parse_known_args()

    ram_path = None
    try:
        # 1. Sync weights to RAM disk
        ram_path = cache_to_ram(args.model, args.shm_dir, args.cache_workers)

        # 2. Prepare vLLM Command
        # Point vLLM to the RAM files, but keep the original model ID for the API
        cmd = [
            "vllm", "serve", str(ram_path),
            "--served-model-name", args.model
        ] + vllm_args

        print(f"\n🔥 Launching vLLM...")
        print(f"   Command: {' '.join(cmd)}\n", file=sys.stderr)
        
        # 3. Run vLLM and wait
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n👋 Process interrupted by user.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ vLLM exited with error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
    finally:
        # 4. Cleanup RAM Disk
        if ram_path and ram_path.exists() and not args.keep_cache:
            print(f"🧹 Cleaning up RAM cache: {ram_path}", file=sys.stderr)
            try:
                shutil.rmtree(ram_path)
                print("✨ RAM disk cleared.", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ Failed to clean {ram_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
