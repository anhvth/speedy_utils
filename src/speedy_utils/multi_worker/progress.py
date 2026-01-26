"""
Real-time progress tracking for distributed Ray tasks.

This module provides a ProgressActor that allows workers to report item-level
progress in real-time, giving users visibility into actual items processed
rather than just task completion.
"""
import time
import threading
from typing import Optional, Callable

__all__ = ['ProgressActor', 'create_progress_tracker', 'get_ray_progress_actor']


def get_ray_progress_actor():
    """Get the Ray-decorated ProgressActor class (lazy import to avoid Ray at module load)."""
    import ray
    
    @ray.remote
    class ProgressActor:
        """
        A Ray actor for tracking real-time progress across distributed workers.
        
        Workers call `update(n)` to report items processed, and the main process
        can poll `get_progress()` to update a tqdm bar in real-time.
        """
        def __init__(self, total: int, desc: str = "Items"):
            self.total = total
            self.processed = 0
            self.desc = desc
            self.start_time = time.time()
            self._lock = threading.Lock()
        
        def update(self, n: int = 1) -> int:
            """Increment processed count by n. Returns new total."""
            with self._lock:
                self.processed += n
                return self.processed
        
        def get_progress(self) -> dict:
            """Get current progress stats."""
            with self._lock:
                elapsed = time.time() - self.start_time
                rate = self.processed / elapsed if elapsed > 0 else 0
                return {
                    "processed": self.processed,
                    "total": self.total,
                    "elapsed": elapsed,
                    "rate": rate,
                    "desc": self.desc,
                }
        
        def set_total(self, total: int):
            """Update total (useful if exact count unknown at start)."""
            with self._lock:
                self.total = total
        
        def reset(self):
            """Reset progress counter."""
            with self._lock:
                self.processed = 0
                self.start_time = time.time()
    
    return ProgressActor


def create_progress_tracker(total: int, desc: str = "Items"):
    """
    Create a progress tracker actor for use with Ray distributed tasks.
    
    Args:
        total: Total number of items to process
        desc: Description for the progress bar
        
    Returns:
        A Ray actor handle that workers can use to report progress
        
    Example:
        progress_actor = create_progress_tracker(1000000, "Processing items")
        
        @ray.remote
        def worker(items, progress_actor):
            for item in items:
                process(item)
                ray.get(progress_actor.update.remote(1))
        
        # In main process, poll progress:
        while not done:
            stats = ray.get(progress_actor.get_progress.remote())
            pbar.n = stats["processed"]
            pbar.refresh()
    """
    import ray
    ProgressActor = get_ray_progress_actor()
    return ProgressActor.remote(total, desc)


class ProgressPoller:
    """
    Background thread that polls a Ray progress actor and updates a tqdm bar.
    """
    def __init__(self, progress_actor, pbar, poll_interval: float = 0.5):
        import ray
        self._ray = ray
        self.progress_actor = progress_actor
        self.pbar = pbar
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the polling thread."""
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the polling thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _poll_loop(self):
        """Poll the progress actor and update tqdm."""
        while not self._stop_event.is_set():
            try:
                stats = self._ray.get(self.progress_actor.get_progress.remote())
                self.pbar.n = stats["processed"]
                self.pbar.set_postfix_str(f'{stats["rate"]:.1f} items/s')
                self.pbar.refresh()
            except Exception:
                pass  # Ignore errors during polling
            self._stop_event.wait(self.poll_interval)
        
        # Final update
        try:
            stats = self._ray.get(self.progress_actor.get_progress.remote())
            self.pbar.n = stats["processed"]
            self.pbar.refresh()
        except Exception:
            pass
