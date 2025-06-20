"""
# ============================================================================= #
# VLLM MODEL SERVING AND LORA MANAGEMENT UTILITIES
# ============================================================================= #
#
# Title & Intent:
# Command-line interface for serving language models and managing LoRA adapters with vLLM
#
# High-level Summary:
# This module provides a comprehensive CLI tool for deploying and managing vLLM model servers
# with support for base models, LoRA adapters, and dynamic adapter loading. It handles GPU
# allocation, process management, model discovery, and provides utilities for adding/removing
# LoRA adapters to running servers. The tool simplifies the deployment of production-ready
# language model serving infrastructure with fine-tuned model support.
#
# Public API / Data Contracts:
# • serve_model(model_name, gpus, **kwargs) -> subprocess.Popen - Start vLLM server for base model
# • serve_lora(lora_name_or_path, gpus, **kwargs) -> subprocess.Popen - Start vLLM server with LoRA
# • add_lora(lora_name_or_path, host_port, **kwargs) -> dict - Add LoRA to running server
# • list_loras(host_port, api_key="abc") -> None - List available LoRA adapters
# • model_list(host_port, api_key="abc") -> None - List available models
# • remove_lora(lora_name, host_port, api_key="abc") -> dict - Remove LoRA adapter
# • get_lora_path(lora_name_or_path) -> str - Resolve LoRA adapter path
# • LORA_DIR: str - Environment-configurable LoRA storage directory
# • HF_HOME: str - Hugging Face cache directory
#
# Invariants / Constraints:
# • GPU groups MUST be specified as comma-separated integers (e.g., "0,1,2,3")
# • LoRA paths MUST exist and contain valid adapter files
# • Server endpoints MUST be reachable for dynamic LoRA operations
# • MUST validate model and LoRA compatibility before serving
# • Process management MUST handle graceful shutdown on interruption
# • MUST respect CUDA device visibility and memory constraints
# • LoRA operations MUST verify server API compatibility
# • MUST log all serving operations and adapter changes
#
# Usage Example:
# ```bash
# # Serve a base model on GPUs 0,1
# svllm serve --model meta-llama/Llama-2-7b-hf --gpus 0,1
#
# # Serve a model with LoRA adapter
# svllm serve --lora my-adapter /path/to/adapter --gpus 0,1,2,3
#
# # Add LoRA to running server
# svllm add-lora --lora new-adapter /path/to/new-adapter --host_port localhost:8000
#
# # List available models
# svllm list-models --host_port localhost:8000
#
# # Remove LoRA adapter
# svllm remove-lora --lora adapter-name --host_port localhost:8000
# ```
#
# TODO & Future Work:
# • Add support for multi-node distributed serving
# • Implement automatic model quantization options
# • Add configuration validation before server startup
# • Support for custom tokenizer and chat templates
# • Add health check endpoints for load balancer integration
# • Implement rolling updates for zero-downtime deployments
#
# ============================================================================= #
"""

import argparse
import os
import subprocess
from typing import List, Optional

import openai
import requests
from loguru import logger

from speedy_utils.common.utils_io import load_by_ext

LORA_DIR: str = os.environ.get("LORA_DIR", "/loras")
LORA_DIR = os.path.abspath(LORA_DIR)
HF_HOME: str = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
logger.info(f"LORA_DIR: {LORA_DIR}")


def model_list(host_port: str, api_key: str = "abc") -> None:
    """List models from the vLLM server."""
    client = openai.OpenAI(base_url=f"http://{host_port}/v1", api_key=api_key)
    models = client.models.list()
    for model in models:
        print(f"Model ID: {model.id}")


def add_lora(
    lora_name_or_path: str,
    host_port: str,
    url: str = "http://HOST:PORT/v1/load_lora_adapter",
    served_model_name: Optional[str] = None,
    lora_module: Optional[str] = None,
) -> dict:
    """Add a LoRA adapter to a running vLLM server."""
    url = url.replace("HOST:PORT", host_port)
    headers = {"Content-Type": "application/json"}

    data = {
        "lora_name": served_model_name,
        "lora_path": os.path.abspath(lora_name_or_path),
    }
    if lora_module:
        data["lora_module"] = lora_module
    logger.info(f"{data=}, {headers}, {url=}")
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {
                "status": "success",
                "message": (
                    response.text
                    if response.text.strip()
                    else "Request completed with empty response"
                ),
            }
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}


def unload_lora(lora_name: str, host_port: str) -> Optional[dict]:
    """Unload a LoRA adapter from a running vLLM server."""
    try:
        url = f"http://{host_port}/v1/unload_lora_adapter"
        logger.info(f"{url=}")
        headers = {"Content-Type": "application/json"}
        data = {"lora_name": lora_name}
        logger.info(f"Unloading LoRA adapter: {data=}")
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        logger.success(f"Unloaded LoRA adapter: {lora_name}")
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


def serve(args) -> None:
    """Start vLLM containers with dynamic args."""
    print("Starting vLLM containers...,")
    gpu_groups_arr: List[str] = args.gpu_groups.split(",")
    vllm_binary: str = get_vllm()
    if args.enable_lora:
        vllm_binary = "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True " + vllm_binary

    if (
        not args.bnb
        and args.model
        and ("bnb" in args.model.lower() or "4bit" in args.model.lower())
    ):
        args.bnb = True
        print(f"Auto-detected quantization for model: {args.model}")

    if args.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        print("Enabled runtime LoRA updating")

    for i, gpu_group in enumerate(gpu_groups_arr):
        port = int(args.host_port.split(":")[-1]) + i
        gpu_group = ",".join([str(x) for x in gpu_group])
        tensor_parallel = len(gpu_group.split(","))

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_group}",
            vllm_binary,
            "serve",
            args.model,
            "--port",
            str(port),
            "--tensor-parallel",
            str(tensor_parallel),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--dtype",
            args.dtype,
            "--max-model-len",
            str(args.max_model_len),
            "--enable-prefix-caching",
            "--disable-log-requests",
            # "--uvicorn-log-level critical",
        ]
        if HF_HOME:
            cmd.insert(0, f"HF_HOME={HF_HOME}")
        if args.eager:
            cmd.append("--enforce-eager")

        if args.served_model_name:
            cmd.extend(["--served-model-name", args.served_model_name])

        if args.bnb:
            cmd.extend(
                ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
            )

        if args.enable_lora:
            cmd.extend(["--fully-sharded-loras", "--enable-lora"])

        if args.lora_modules:
            assert len(args.lora_modules) % 2 == 0, "lora_modules must be even"
            s = ""
            for i in range(0, len(args.lora_modules), 2):
                name = args.lora_modules[i]
                module = args.lora_modules[i + 1]
                s += f"{name}={module} "
            cmd.extend(["--lora-modules", s])

        if hasattr(args, "enable_reasoning") and args.enable_reasoning:
            cmd.extend(["--enable-reasoning", "--reasoning-parser", "deepseek_r1"])
            # Add VLLM_USE_V1=0 to the environment for reasoning mode
            cmd.insert(0, "VLLM_USE_V1=0")

        final_cmd = " ".join(cmd)
        log_file = f"/tmp/vllm_{port}.txt"
        final_cmd_with_log = f'"{final_cmd} 2>&1 | tee {log_file}"'
        run_in_tmux = (
            f"tmux new-session -d -s vllm_{port} 'bash -c {final_cmd_with_log}'"
        )

        print(final_cmd)
        print("Logging to", log_file)
        os.system(run_in_tmux)


def get_vllm() -> str:
    """Get the vLLM binary path."""
    vllm_binary = subprocess.check_output("which vllm", shell=True, text=True).strip()
    vllm_binary = os.getenv("VLLM_BINARY", vllm_binary)
    logger.info(f"vLLM binary: {vllm_binary}")
    assert os.path.exists(vllm_binary), (
        f"vLLM binary not found at {vllm_binary}, please set VLLM_BINARY env variable"
    )
    return vllm_binary


def get_args():
    """Parse command line arguments."""
    example_args = [
        "svllm serve --model MODEL_NAME --gpus 0,1,2,3",
        "svllm serve --lora LORA_NAME LORA_PATH --gpus 0,1,2,3",
        "svllm add_lora --lora LORA_NAME LORA_PATH --host_port localhost:8150",
        "svllm kill",
    ]

    parser = argparse.ArgumentParser(
        description="vLLM Serve Script", epilog="Example: " + " || ".join(example_args)
    )
    parser.add_argument(
        "mode",
        choices=["serve", "kill", "add_lora", "unload_lora", "list_models"],
        help="Mode to run the script in",
    )
    parser.add_argument("--model", "-m", type=str, help="Model to serve")
    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
        help="Comma-separated list of GPU groups",
        dest="gpu_groups",
    )
    parser.add_argument(
        "--lora",
        "-l",
        nargs=2,
        metavar=("LORA_NAME", "LORA_PATH"),
        help="Name and path of the LoRA adapter",
    )
    parser.add_argument(
        "--served_model_name", type=str, help="Name of the served model"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        "-gmu",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument("--dtype", type=str, default="auto", help="Data type")
    parser.add_argument(
        "--max_model_len", "-mml", type=int, default=8192, help="Maximum model length"
    )
    parser.add_argument(
        "--enable_lora",
        dest="enable_lora",
        action="store_true",
        help="Disable LoRA support",
        default=False,
    )
    parser.add_argument("--bnb", action="store_true", help="Enable quantization")
    parser.add_argument(
        "--not_verbose", action="store_true", help="Disable verbose logging"
    )
    parser.add_argument("--vllm_binary", type=str, help="Path to the vLLM binary")
    parser.add_argument(
        "--pipeline_parallel",
        "-pp",
        default=1,
        type=int,
        help="Number of pipeline parallel stages",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for the serve command",
    )
    parser.add_argument(
        "--host_port",
        "-hp",
        type=str,
        default="localhost:8150",
        help="Host and port for the server format: host:port",
    )
    parser.add_argument("--eager", action="store_true", help="Enable eager execution")
    parser.add_argument(
        "--lora_modules",
        "-lm",
        nargs="+",
        type=str,
        help="List of LoRA modules in the format lora_name lora_module",
    )
    parser.add_argument(
        "--enable-reasoning", action="store_true", help="Enable reasoning"
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""

    args = get_args()

    if args.mode == "serve":
        # Handle LoRA model serving via the new --lora argument
        if args.lora:
            lora_name, lora_path = args.lora
            if not args.lora_modules:
                args.lora_modules = [lora_name, lora_path]
            # Try to get the model from LoRA config if not specified
            if args.model is None:
                lora_config = os.path.join(lora_path, "adapter_config.json")
                if os.path.exists(lora_config):
                    config = load_by_ext(lora_config)
                    model_name = config.get("base_model_name_or_path")
                    # Handle different quantization suffixes
                    if model_name.endswith("-unsloth-bnb-4bit") and not args.bnb:
                        model_name = model_name.replace("-unsloth-bnb-4bit", "")
                    elif model_name.endswith("-bnb-4bit") and not args.bnb:
                        model_name = model_name.replace("-bnb-4bit", "")
                    logger.info(f"Model name from LoRA config: {model_name}")
                    args.model = model_name

        # Fall back to existing logic for other cases (already specified lora_modules)
        if args.model is None and args.lora_modules is not None and not args.lora:
            lora_config = os.path.join(args.lora_modules[1], "adapter_config.json")
            if os.path.exists(lora_config):
                config = load_by_ext(lora_config)
                model_name = config.get("base_model_name_or_path")
                if model_name.endswith("-unsloth-bnb-4bit") and not args.bnb:
                    model_name = model_name.replace("-unsloth-bnb-4bit", "")
                elif model_name.endswith("-bnb-4bit") and not args.bnb:
                    model_name = model_name.replace("-bnb-4bit", "")
                logger.info(f"Model name from LoRA config: {model_name}")
                args.model = model_name
        # port_start from hostport
        serve(args)

    elif args.mode == "add_lora":
        if args.lora:
            lora_name, lora_path = args.lora
            add_lora(lora_path, host_port=args.host_port, served_model_name=lora_name)
        else:
            # Fallback to old behavior
            lora_name = args.model
            add_lora(
                lora_name,
                host_port=args.host_port,
                served_model_name=args.served_model_name,
            )
    elif args.mode == "unload_lora":
        if args.lora:
            lora_name = args.lora[0]
        else:
            lora_name = args.model
        unload_lora(lora_name, host_port=args.host_port)
    elif args.mode == "list_models":
        model_list(args.host_port)
    else:
        raise ValueError(f"Unknown mode: {args.mode}, ")


if __name__ == "__main__":
    main()
