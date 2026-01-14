import time
import random
import ray
from vllm import LLM, SamplingParams
from speedy_utils.multi_worker.parallel_gpu_pool import RayWorkerBase, RayRunner
import os
ray.init(ignore_reinit_error=True)

# --- Define Your Worker ---
class MyEduWorker(RayWorkerBase):
    def setup(self):
        print(f"Worker {self.worker_id}: Loading vLLM Engine...")
        
        # Initialize vLLM
        # Note: Set gpu_memory_utilization based on how many workers share a GPU
        self.model = LLM(
            model="Qwen/Qwen3-0.6B", 
            gpu_memory_utilization=0.4, # Adjust based on your GPU pool density
            trust_remote_code=True, 
            enforce_eager=True,
            
        )
        
        # Set default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
    def process_one_item(self, item):
        # 'item' is the prompt from your all_files list
        prompt = f"Summarize this file metadata: {item}"
        
        # vLLM offline generation
        outputs = self.model.generate([prompt], self.sampling_params)
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        
        return {
            "file": item,
            "response": generated_text.strip(),
            "worker_id": self.worker_id,
            "gpu_idx": ray.get_runtime_context().get_assigned_resources().get("GPU", []),
            "node_id": ray.get_runtime_context().node_id.hex(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")
        }

# --- Run It ---
# Create fake data (prompts or filenames)
all_files = [f"document_id_{i}" for i in range(20)]

# Set test_mode=False if you want to use real GPUs
runner = RayRunner(test_mode=False, gpus_per_worker=2)
results = runner.run(
    worker_class=MyEduWorker,
    all_data=all_files
)
from speedy_utils import dump_json_or_pickle
dump_json_or_pickle(results, "edu_results.json")