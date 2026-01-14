import os
import sys
import ray
import time
import datetime
import numpy as np
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

# --- 1. Shared Global Counter (Ray Actor) ---
@ray.remote
class ProgressTracker:
    def __init__(self, total_items):
        self.total_items = total_items
        self.processed_count = 0
        self.start_time = time.time()
        
    def increment(self):
        self.processed_count += 1
        
    def get_stats(self):
        elapsed = time.time() - self.start_time
        speed = self.processed_count / elapsed if elapsed > 0 else 0
        return self.processed_count, self.total_items, speed, elapsed

# --- 2. Cluster Manager ---
class RayRunner:
    def __init__(self, gpus_per_worker=1, test_mode=False):
        self.gpus_per_worker = gpus_per_worker
        self.test_mode = test_mode
        
        # Logging Setup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base = f"/tmp/raylog/runs_{timestamp}"
        
        # Initialize Ray if not in test mode
        if self.test_mode:
            print(f">>> [TEST MODE] Running locally (CPU).")
        else:
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            
            resources = ray.cluster_resources()
            self.total_gpus = int(resources.get("GPU", 0))
            if self.total_gpus == 0:
                raise RuntimeError("No GPUs found in cluster!")
                
            print(f">>> Connected. Available GPUs: {self.total_gpus}")
            print(f">>> Logs redirected to: {self.log_base}")
            os.makedirs(self.log_base, exist_ok=True)

    def run(self, worker_class, all_data, **kwargs):
        # --- TEST MODE ---
        if self.test_mode:
            # Run local simple version
            worker = worker_class(worker_id=0, log_dir=None, tracker=None, **kwargs)
            worker.setup()
            return [worker.process_one_item(x) for x in all_data[:3]]

        # --- CLUSTER MODE ---
        num_workers = self.total_gpus // self.gpus_per_worker
        print(f">>> Spawning {num_workers} workers for {len(all_data)} items.")
        
        # 1. Start the Global Tracker
        tracker = ProgressTracker.remote(len(all_data))

        # 2. Prepare Shards
        shards = np.array_split(all_data, num_workers)
        
        # 3. Create Remote Worker Class
        RemoteWorker = ray.remote(num_gpus=self.gpus_per_worker)(worker_class)
        
        actors = []
        futures = []
        
        for i, shard in enumerate(shards):
            if len(shard) == 0: continue

            # Initialize Actor
            actor = RemoteWorker.remote(
                worker_id=i, 
                log_dir=self.log_base, 
                tracker=tracker, # Pass the tracker handle
                **kwargs
            )
            actors.append(actor)
            
            # Launch Task
            futures.append(actor._run_shard.remote(shard.tolist()))

        results = ray.get(futures)
        return [item for sublist in results for item in sublist]

# --- 3. The Base Worker ---
class RayWorkerBase(ABC):
    def __init__(self, worker_id, log_dir, tracker, **kwargs):
        self.worker_id = worker_id
        self.log_dir = log_dir
        self.tracker = tracker
        self.kwargs = kwargs
        self._log_file_handle = None
        self._last_print_time = 0

    @abstractmethod
    def setup(self):
        """User must override to initialize models/resources"""
        pass

    @abstractmethod
    def process_one_item(self, item):
        """User must override to process a single item"""
        raise NotImplementedError

    def _redirect_output(self):
        """Workers > 0 write to disk. Worker 0 writes to Notebook."""
        if self.worker_id == 0 or self.log_dir is None:
            return

        log_path = os.path.join(self.log_dir, f"worker_{self.worker_id}.log")
        self._log_file_handle = open(log_path, "w", buffering=1)
        sys.stdout = self._log_file_handle
        sys.stderr = self._log_file_handle

    def _print_global_stats(self):
        """Only used by Worker 0 to print pretty global stats"""
        if self.tracker is None: return
        
        # Limit print frequency to every 5 seconds to avoid spamming Jupyter
        if time.time() - self._last_print_time < 5:
            return

        # Fetch stats from the Actor
        count, total, speed, elapsed = ray.get(self.tracker.get_stats.remote())
        
        if speed > 0:
            eta = (total - count) / speed
            eta_str = str(datetime.timedelta(seconds=int(eta)))
        else:
            eta_str = "?"

        # \r allows overwriting the line (basic animation)
        msg = (f"[Global] {count}/{total} | {count/total:.1%} | "
               f"Speed: {speed:.2f} it/s | ETA: {eta_str}")
        print(msg)
        self._last_print_time = time.time()

    def _run_shard(self, shard):
        self._redirect_output()
        try:
            self.setup()
            results = []
            
            # Simple loop, no tqdm needed for Worker 0 as it prints Global Stats
            # Worker > 0 can use tqdm if they want, but it goes to log file
            iterator = shard
            if self.worker_id > 0:
                iterator = tqdm(shard, desc=f"Worker {self.worker_id}")

            for item in iterator:
                try:
                    res = self.process_one_item(item)
                    results.append(res)
                except Exception as e:
                    print(f"Error {item}: {e}")
                    results.append(None)
                
                # Update Global Counter
                if self.tracker:
                    self.tracker.increment.remote()

                # Worker 0: Print Global Stats
                if self.worker_id == 0:
                    self._print_global_stats()
            
            return results
        finally:
            if self._log_file_handle:
                self._log_file_handle.close()