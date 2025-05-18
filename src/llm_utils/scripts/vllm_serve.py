""" "
USAGE:
Serve models and LoRAs with vLLM:

Serve a LoRA model:
svllm serve --lora LORA_NAME LORA_PATH --gpus GPU_GROUPS

Serve a base model:
svllm serve --model MODEL_NAME --gpus GPU_GROUPS

Add a LoRA to a served model:
svllm add-lora --lora LORA_NAME LORA_PATH --host_port host:port (if add then the port must be specify)
"""

from glob import glob
import os
import subprocess
import time
from typing import List, Literal, Optional
from fastcore.script import call_parse
from loguru import logger
import argparse
import requests
import openai


LORA_DIR = os.environ.get("LORA_DIR", "/loras")
LORA_DIR = os.path.abspath(LORA_DIR)
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
logger.info(f"LORA_DIR: {LORA_DIR}")


def model_list(host_port, api_key="abc"):
    client = openai.OpenAI(base_url=f"http://{host_port}/v1", api_key=api_key)
    models = client.models.list()
    for model in models:
        print(f"Model ID: {model.id}")


def kill_existing_vllm(vllm_binary: Optional[str] = None) -> None:
    """Kill selected vLLM processes using fzf."""
    if not vllm_binary:
        vllm_binary = get_vllm()

    # List running vLLM processes
    result = subprocess.run(
        f"ps aux | grep {vllm_binary} | grep -v grep",
        shell=True,
        capture_output=True,
        text=True,
    )
    processes = result.stdout.strip().split("\n")

    if not processes or processes == [""]:
        print("No running vLLM processes found.")
        return

    # Use fzf to select processes to kill
    fzf = subprocess.Popen(
        ["fzf", "--multi"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    selected, _ = fzf.communicate("\n".join(processes))

    if not selected:
        print("No processes selected.")
        return

    # Extract PIDs and kill selected processes
    pids = [line.split()[1] for line in selected.strip().split("\n")]
    for pid in pids:
        subprocess.run(
            f"kill -9 {pid}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"Killed processes: {', '.join(pids)}")


def add_lora(
    lora_name_or_path: str,
    host_port: str,
    url: str = "http://HOST:PORT/v1/load_lora_adapter",
    served_model_name: Optional[str] = None,
    lora_module: Optional[str] = None,  # Added parameter
) -> dict:
    url = url.replace("HOST:PORT", host_port)
    headers = {"Content-Type": "application/json"}

    data = {
        "lora_name": served_model_name,
        "lora_path": os.path.abspath(lora_name_or_path),
    }
    if lora_module:  # Include lora_module if provided
        data["lora_module"] = lora_module
    logger.info(f"{data=}, {headers}, {url=}")
    # logger.warning(f"Failed to unload LoRA adapter: {str(e)}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        # Handle potential non-JSON responses
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


def unload_lora(lora_name, host_port):
    try:
        url = f"http://{host_port}/v1/unload_lora_adapter"
        logger.info(f"{url=}")
        headers = {"Content-Type": "application/json"}
        data = {"lora_name": lora_name}
        logger.info(f"Unloading LoRA adapter: {data=}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.success(f"Unloaded LoRA adapter: {lora_name}")
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


def serve(
    model: str,
    gpu_groups: str,
    served_model_name: Optional[str] = None,
    port_start: int = 8155,
    gpu_memory_utilization: float = 0.93,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
    enable_lora: bool = False,
    is_bnb: bool = False,
    eager: bool = False,
    chat_template: Optional[str] = None,
    lora_modules: Optional[List[str]] = None,  # Updated type
):
    """Main function to start or kill vLLM containers."""

    """Start vLLM containers with dynamic args."""
    print("Starting vLLM containers...,")
    gpu_groups_arr = gpu_groups.split(",")
    VLLM_BINARY = get_vllm()
    if enable_lora:
        VLLM_BINARY = "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True " + VLLM_BINARY

    # Auto-detect quantization based on model name if not explicitly set
    if not is_bnb and model and ("bnb" in model.lower() or "4bit" in model.lower()):
        is_bnb = True
        print(f"Auto-detected quantization for model: {model}")

    # Set environment variables for LoRA if needed
    if enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        print("Enabled runtime LoRA updating")

    for i, gpu_group in enumerate(gpu_groups_arr):
        port = port_start + i
        gpu_group = ",".join([str(x) for x in gpu_group])
        tensor_parallel = len(gpu_group.split(","))

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_group}",
            VLLM_BINARY,
            "serve",
            model,
            "--port",
            str(port),
            "--tensor-parallel",
            str(tensor_parallel),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--dtype",
            dtype,
            "--max-model-len",
            str(max_model_len),
            "--enable-prefix-caching",
            "--disable-log-requests",
            "--uvicorn-log-level critical",
        ]
        if HF_HOME:
            # insert
            cmd.insert(0, f"HF_HOME={HF_HOME}")
        if eager:
            cmd.append("--enforce-eager")

        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])

        if is_bnb:
            cmd.extend(
                ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
            )

        if enable_lora:
            cmd.extend(["--fully-sharded-loras", "--enable-lora"])

        if chat_template:
            chat_template = get_chat_template(chat_template)
            cmd.extend(["--chat-template", chat_template])  # Add chat_template argument
        if lora_modules:
            # for lora_module in lora_modules:
            # len must be even and we will join tuple with `=`
            assert len(lora_modules) % 2 == 0, "lora_modules must be even"
            # lora_modulle = [f'{name}={module}' for name, module in zip(lora_module[::2], lora_module[1::2])]
            # import ipdb;ipdb.set_trace()
            s = ""
            for i in range(0, len(lora_modules), 2):
                name = lora_modules[i]
                module = lora_modules[i + 1]
                s += f"{name}={module} "

            cmd.extend(["--lora-modules", s])
        # add kwargs
        final_cmd = " ".join(cmd)
        log_file = f"/tmp/vllm_{port}.txt"
        final_cmd_with_log = f'"{final_cmd} 2>&1 | tee {log_file}"'
        run_in_tmux = (
            f"tmux new-session -d -s vllm_{port} 'bash -c {final_cmd_with_log}'"
        )

        print(final_cmd)
        print("Logging to", log_file)
        os.system(run_in_tmux)


def get_vllm():
    VLLM_BINARY = subprocess.check_output("which vllm", shell=True, text=True).strip()
    VLLM_BINARY = os.getenv("VLLM_BINARY", VLLM_BINARY)
    logger.info(f"vLLM binary: {VLLM_BINARY}")
    assert os.path.exists(
        VLLM_BINARY
    ), f"vLLM binary not found at {VLLM_BINARY}, please set VLLM_BINARY env variable"
    return VLLM_BINARY


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
        "--disable_lora",
        dest="enable_lora",
        action="store_false",
        help="Disable LoRA support",
        default=True,
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
    # parser.add_argument(
    #     "--extra_args",
    #     nargs=argparse.REMAINDER,
    #     help="Additional arguments for the serve command",
    # )
    parser.add_argument(
        "--host_port",
        "-hp",
        type=str,
        default="localhost:8150",
        help="Host and port for the server format: host:port",
    )
    parser.add_argument("--eager", action="store_true", help="Enable eager execution")
    parser.add_argument(
        "--chat_template",
        type=str,
        help="Path to the chat template file",
    )
    parser.add_argument(
        "--lora_modules",
        "-lm",
        nargs="+",
        type=str,
        help="List of LoRA modules in the format lora_name lora_module",
    )
    return parser.parse_args()


from speedy_utils import jloads, load_by_ext, memoize


def fetch_chat_template(template_name: str = "qwen") -> str:
    """
    Fetches a chat template file from a remote repository or local cache.

    Args:
        template_name (str): Name of the chat template. Defaults to 'qwen'.

    Returns:
        str: Path to the downloaded or cached chat template file.

    Raises:
        AssertionError: If the template_name is not supported.
        ValueError: If the file URL is invalid.
    """
    supported_templates = [
        "alpaca",
        "chatml",
        "gemma-it",
        "llama-2-chat",
        "mistral-instruct",
        "qwen2.5-instruct",
        "saiga",
        "vicuna",
        "qwen",
    ]
    assert template_name in supported_templates, (
        f"Chat template '{template_name}' not supported. "
        f"Please choose from {supported_templates}."
    )

    # Map 'qwen' to 'qwen2.5-instruct'
    if template_name == "qwen":
        template_name = "qwen2.5-instruct"

    remote_url = (
        f"https://raw.githubusercontent.com/chujiezheng/chat_templates/"
        f"main/chat_templates/{template_name}.jinja"
    )
    local_cache_path = f"/tmp/chat_template_{template_name}.jinja"

    if remote_url.startswith("http"):
        import requests

        response = requests.get(remote_url)
        with open(local_cache_path, "w") as file:
            file.write(response.text)
        return local_cache_path

    raise ValueError("The file URL must be a valid HTTP URL.")


def get_chat_template(template_name: str) -> str:
    return fetch_chat_template(template_name)


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
        port_start = int(args.host_port.split(":")[-1])
        serve(
            args.model,
            args.gpu_groups,
            args.served_model_name,
            port_start,
            args.gpu_memory_utilization,
            args.dtype,
            args.max_model_len,
            args.enable_lora,
            args.bnb,
            args.eager,
            args.chat_template,
            args.lora_modules,
        )

    elif args.mode == "kill":
        kill_existing_vllm(args.vllm_binary)
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
