"""
LLMRay: Simplified Ray-based vLLM wrapper for offline batch inference.

Automatically handles data parallelism across available GPUs in Ray cluster.
Pipeline parallel is always 1 (no layer splitting).

Example:
    # dp=4, tp=2 means 8 GPUs total, 4 model replicas each using 2 GPUs
    llm = LLMRay(model_name='Qwen/Qwen3-0.6B', dp=4, tp=2)

    # dp=8, tp=2 means 16 GPUs across nodes, 8 model replicas
    llm = LLMRay(model_name='meta-llama/Llama-3-70B', dp=8, tp=2)
"""
import os
import datetime
import ray
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from tqdm.auto import tqdm

# Type alias for OpenAI-style messages
Message = Dict[str, str]  # {'role': str, 'content': str}
Messages = List[Message]


@ray.remote
class _ProgressTracker:
    """Ray actor for tracking global progress across workers."""

    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_count = 0
        import time
        self.start_time = time.time()

    def increment(self) -> None:
        self.processed_count += 1

    def get_stats(self) -> tuple:
        import time
        elapsed = time.time() - self.start_time
        speed = self.processed_count / elapsed if elapsed > 0 else 0
        return self.processed_count, self.total_items, speed, elapsed


class _VLLMWorkerBase(ABC):
    """Base worker class for vLLM inference."""

    def __init__(
        self,
        worker_id: int,
        log_dir: Optional[str],
        tracker: Any,
        **kwargs: Any,
    ):
        self.worker_id = worker_id
        self.log_dir = log_dir
        self.tracker = tracker
        self.kwargs = kwargs
        self._log_file_handle = None
        self._last_print_time = 0

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def process_one_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def _redirect_output(self) -> None:
        """Workers > 0 write to disk. Worker 0 writes to stdout."""
        import sys
        if self.worker_id == 0 or self.log_dir is None:
            return
        log_path = os.path.join(self.log_dir, f'worker_{self.worker_id}.log')
        self._log_file_handle = open(log_path, 'w', buffering=1)
        sys.stdout = self._log_file_handle
        sys.stderr = self._log_file_handle

    def _print_global_stats(self) -> None:
        """Only used by Worker 0 to print global stats."""
        import time
        import datetime as dt
        if self.tracker is None:
            return
        if time.time() - self._last_print_time < 5:
            return
        count, total, speed, elapsed = ray.get(self.tracker.get_stats.remote())
        if speed > 0:
            eta = (total - count) / speed
            eta_str = str(dt.timedelta(seconds=int(eta)))
        else:
            eta_str = '?'
        msg = (
            f'[Global] {count}/{total} | {count/total:.1%} | '
            f'Speed: {speed:.2f} it/s | ETA: {eta_str}'
        )
        print(msg)
        self._last_print_time = time.time()

    def _run_shard(self, shard: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._redirect_output()
        try:
            self.setup()
            results = []
            iterator = shard
            if self.worker_id > 0:
                iterator = tqdm(shard, desc=f'Worker {self.worker_id}')
            for item in iterator:
                try:
                    res = self.process_one_item(item)
                    results.append(res)
                except Exception as e:
                    print(f'Error {item}: {e}')
                    results.append(None)
                if self.tracker:
                    self.tracker.increment.remote()
                if self.worker_id == 0:
                    self._print_global_stats()
            return results
        finally:
            if self._log_file_handle:
                self._log_file_handle.close()


class _VLLMWorker(_VLLMWorkerBase):
    """Worker that runs vLLM inference on assigned GPUs."""

    def setup(self) -> None:
        """Initialize vLLM engine with configured parameters."""
        from vllm import LLM

        model_name = self.kwargs['model_name']
        tp = self.kwargs.get('tp', 1)
        gpu_memory_utilization = self.kwargs.get(
            'gpu_memory_utilization', 0.9
        )
        trust_remote_code = self.kwargs.get('trust_remote_code', True)
        vllm_kwargs = self.kwargs.get('vllm_kwargs', {})

        print(
            f'Worker {self.worker_id}: Loading vLLM model {model_name} '
            f'with TP={tp}...'
        )

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,  # Always 1 as per requirement
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enforce_eager=True,
            **vllm_kwargs,
        )

        # Store default sampling params
        self.default_sampling_params = self.kwargs.get(
            'sampling_params', {}
        )

    def process_one_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single input item with OpenAI-style messages."""
        from vllm import SamplingParams

        messages = item.get('messages')
        if not messages:
            raise ValueError('Item must contain "messages" key')

        # Validate messages format
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(
                    f'Each message must be dict, got {type(msg)}'
                )
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(
                    'Each message must have "role" and "content"'
                )

        # Build sampling params (item-specific overrides default)
        sampling_config = {
            **self.default_sampling_params,
            **item.get('sampling_params', {}),
        }
        sampling_params = SamplingParams(**sampling_config)

        # Use vLLM chat interface
        outputs = self.model.chat(
            messages=[messages],
            sampling_params=sampling_params,
        )
        generated_text = outputs[0].outputs[0].text

        # Build result
        result = {
            'messages': messages,
            'generated_text': generated_text,
            'worker_id': self.worker_id,
            'finish_reason': outputs[0].outputs[0].finish_reason,
        }

        # Include any extra metadata from input
        for key in item:
            if key not in ['messages', 'sampling_params']:
                result[f'meta_{key}'] = item[key]

        return result


class LLMRay:
    """
    Ray-based LLM wrapper for offline batch inference with OpenAI messages.

    Spawns multiple model replicas (data parallel) across GPUs/nodes.
    Each replica can use multiple GPUs (tensor parallel).

    Args:
        model_name: HuggingFace model name or path
        dp: Data parallel - number of model replicas
        tp: Tensor parallel - GPUs per replica
        Total GPUs used = dp * tp

    Example:
        # 8 GPUs: 4 replicas, each using 2 GPUs
        >>> llm = LLMRay(model_name='Qwen/Qwen3-0.6B', dp=4, tp=2)

        # 16 GPUs across 2 nodes: 8 replicas, each using 2 GPUs
        >>> llm = LLMRay(model_name='meta-llama/Llama-3-70B', dp=8, tp=2)

        >>> inputs = [
        ...     [{'role': 'user', 'content': 'What is AI?'}],
        ...     [{'role': 'user', 'content': 'Explain quantum computing.'}],
        ... ]
        >>> results = llm.generate(inputs)
    """

    def __init__(
        self,
        model_name: str,
        dp: int = 1,
        tp: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        sampling_params: Optional[Dict[str, Any]] = None,
        vllm_kwargs: Optional[Dict[str, Any]] = None,
        ray_address: Optional[str] = None,
    ):
        """
        Initialize LLMRay.

        Args:
            model_name: HuggingFace model name or path
            dp: Data parallel - number of model replicas (workers)
            tp: Tensor parallel - number of GPUs per replica
            gpu_memory_utilization: Fraction of GPU memory to use
            trust_remote_code: Whether to trust remote code from HF
            sampling_params: Default sampling parameters
            vllm_kwargs: Additional kwargs to pass to vLLM constructor
            ray_address: Ray cluster address ('auto' for existing cluster,
                None for local, or specific address like 'ray://...')
        """
        self.model_name = model_name
        self.dp = dp
        self.tp = tp
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.sampling_params = sampling_params or {
            'temperature': 0.7,
            'max_tokens': 512,
        }
        self.vllm_kwargs = vllm_kwargs or {}
        self.ray_address = ray_address

        # Setup logging
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_base = f'/tmp/raylog/llmray_{timestamp}'

        # Initialize Ray
        self._init_ray()

    def _init_ray(self) -> None:
        """Initialize Ray cluster connection."""
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)

        resources = ray.cluster_resources()
        total_gpus = int(resources.get('GPU', 0))
        required_gpus = self.dp * self.tp

        if total_gpus == 0:
            raise RuntimeError('No GPUs found in Ray cluster!')

        if total_gpus < required_gpus:
            raise RuntimeError(
                f'Not enough GPUs: need {required_gpus} (dp={self.dp} x '
                f'tp={self.tp}), but cluster has {total_gpus}'
            )

        print(f'>>> Ray cluster connected. Total GPUs: {total_gpus}')
        print(f'>>> Config: dp={self.dp}, tp={self.tp} → {required_gpus} GPUs')
        print(f'>>> Logs: {self.log_base}')
        os.makedirs(self.log_base, exist_ok=True)

    def generate(self, inputs: List[Messages]) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of message lists.

        Args:
            inputs: List of message lists, where each message list is
                OpenAI-style: [{'role': 'user', 'content': '...'}]

        Returns:
            List of result dictionaries with generated text and metadata
        """
        # Normalize inputs to dict format with 'messages' key
        normalized_inputs = []
        for messages in inputs:
            if not isinstance(messages, list):
                raise ValueError(
                    f'Each input must be list of messages, got {type(messages)}'
                )
            normalized_inputs.append({'messages': messages})

        num_workers = self.dp
        print(f'>>> Spawning {num_workers} workers for {len(inputs)} items.')

        # 1. Start the Global Tracker
        tracker = _ProgressTracker.remote(len(normalized_inputs))

        # 2. Prepare Shards
        shards = np.array_split(normalized_inputs, num_workers)

        # 3. Create Remote Worker Class with tp GPUs per worker
        RemoteWorker = ray.remote(num_gpus=self.tp)(_VLLMWorker)

        actors = []
        futures = []

        for i, shard in enumerate(shards):
            if len(shard) == 0:
                continue

            # Initialize Actor
            actor = RemoteWorker.remote(
                worker_id=i,
                log_dir=self.log_base,
                tracker=tracker,
                model_name=self.model_name,
                tp=self.tp,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                sampling_params=self.sampling_params,
                vllm_kwargs=self.vllm_kwargs,
            )
            actors.append(actor)

            # Launch Task
            futures.append(actor._run_shard.remote(shard.tolist()))

        results = ray.get(futures)
        return [item for sublist in results for item in sublist]

    def __call__(self, inputs: List[Messages]) -> List[Dict[str, Any]]:
        """Alias for generate()."""
        return self.generate(inputs)
