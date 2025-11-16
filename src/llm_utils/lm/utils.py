import os

# Additional imports for VLLM utilities
import re
import signal
import subprocess
import time
from typing import Any, List, Optional, cast

import requests
from loguru import logger
from openai import OpenAI


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = cast(Any, None)
    logger.warning(
        "psutil not available. Some VLLM process management features may be limited."
    )

# Global tracking of VLLM processes
_VLLM_PROCESSES: list[subprocess.Popen] = []


def _extract_port_from_vllm_cmd(vllm_cmd: str) -> int:
    """Extract port from VLLM command string."""
    port_match = re.search(r"--port\s+(\d+)", vllm_cmd)
    if port_match:
        return int(port_match.group(1))
    return 8000


def _parse_env_vars_from_cmd(cmd: str) -> tuple[dict[str, str], str]:
    """Parse environment variables from command string.

    Args:
        cmd: Command string that may contain environment variables like 'VAR=value command...'

    Returns:
        Tuple of (env_dict, cleaned_cmd) where env_dict contains parsed env vars
        and cleaned_cmd is the command without the env vars.
    """
    import shlex

    # Split the command while preserving quoted strings
    parts = shlex.split(cmd)

    env_vars = {}
    cmd_parts = []

    for part in parts:
        if "=" in part and not part.startswith("-"):
            # Check if this looks like an environment variable
            # Should be KEY=VALUE format, not contain spaces (unless quoted), and KEY should be uppercase
            key_value = part.split("=", 1)
            if len(key_value) == 2:
                key, value = key_value
                if key.isupper() and key.replace("_", "").isalnum():
                    env_vars[key] = value
                    continue

        # Not an env var, add to command parts
        cmd_parts.append(part)

    # Reconstruct the cleaned command
    cleaned_cmd = " ".join(cmd_parts)

    return env_vars, cleaned_cmd


def _start_vllm_server(vllm_cmd: str, timeout: int = 120) -> subprocess.Popen:
    """Start VLLM server and wait for ready."""
    # Parse environment variables from command
    env_vars, cleaned_cmd = _parse_env_vars_from_cmd(vllm_cmd)

    port = _extract_port_from_vllm_cmd(cleaned_cmd)

    logger.info(f"Starting VLLM server: {cleaned_cmd}")
    if env_vars:
        logger.info(f"Environment variables: {env_vars}")
    logger.info(f"VLLM output logged to: /tmp/vllm_{port}.txt")

    with open(f"/tmp/vllm_{port}.txt", "w") as log_file:
        log_file.write(f"VLLM Server started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Command: {cleaned_cmd}\n")
        if env_vars:
            log_file.write(f"Environment: {env_vars}\n")
        log_file.write(f"Port: {port}\n")
        log_file.write("-" * 50 + "\n")

    # Prepare environment for subprocess
    env = os.environ.copy()
    env.update(env_vars)

    with open(f"/tmp/vllm_{port}.txt", "a") as log_file:
        process = subprocess.Popen(
            cleaned_cmd.split(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
            env=env,
        )

    _VLLM_PROCESSES.append(process)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"VLLM server ready on port {port}")
                return process
        except requests.RequestException:
            pass

        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"VLLM server terminated unexpectedly. Return code: {process.returncode}, stderr: {stderr[:200]}..."
            )

        time.sleep(2)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    if process in _VLLM_PROCESSES:
        _VLLM_PROCESSES.remove(process)

    raise RuntimeError(f"VLLM server failed to start within {timeout}s on port {port}")


def _kill_vllm_on_port(port: int) -> bool:
    """Kill VLLM server on port."""
    killed = False
    logger.info(f"Checking VLLM server on port {port}")

    processes_to_remove = []
    for process in _VLLM_PROCESSES:
        try:
            if process.poll() is None:
                killed_process = False
                if HAS_PSUTIL:
                    try:
                        proc = psutil.Process(process.pid)
                        cmdline = " ".join(proc.cmdline())
                        if f"--port {port}" in cmdline or f"--port={port}" in cmdline:
                            logger.info(
                                f"Killing tracked VLLM process {process.pid} on port {port}"
                            )
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                                process.wait()
                            killed = True
                            killed_process = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if not HAS_PSUTIL or not killed_process:
                    logger.info(f"Killing tracked VLLM process {process.pid}")
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            process.wait()
                        killed = True
                    except (ProcessLookupError, OSError):
                        pass

                processes_to_remove.append(process)
            else:
                processes_to_remove.append(process)
        except (ProcessLookupError, OSError):
            processes_to_remove.append(process)

    for process in processes_to_remove:
        if process in _VLLM_PROCESSES:
            _VLLM_PROCESSES.remove(process)

    if not killed and HAS_PSUTIL:
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    if "vllm" in cmdline.lower() and (
                        f"--port {port}" in cmdline or f"--port={port}" in cmdline
                    ):
                        logger.info(
                            f"Killing untracked VLLM process {proc.info['pid']} on port {port}"
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        killed = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Error searching processes on port {port}: {e}")

    if killed:
        logger.info(f"Killed VLLM server on port {port}")
        time.sleep(2)
    else:
        logger.info(f"No VLLM server on port {port}")

    return killed


def stop_vllm_process(process: subprocess.Popen, wait_timeout: int = 10) -> None:
    """Terminate a tracked VLLM process and remove it from tracking."""
    logger.info(f"Stopping VLLM process {process.pid}")
    try:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=wait_timeout)
                logger.info("VLLM process stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("VLLM process didn't stop gracefully, forcing kill")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
        else:
            logger.info("VLLM process already terminated")
    except (ProcessLookupError, OSError) as exc:
        logger.warning(f"Process may have already terminated: {exc}")
    finally:
        if process in _VLLM_PROCESSES:
            _VLLM_PROCESSES.remove(process)


def kill_all_vllm_processes() -> int:
    """Kill all tracked VLLM processes."""
    killed_count = 0
    for process in list(_VLLM_PROCESSES):
        if process.poll() is None:
            logger.info(f"Killing VLLM process with PID {process.pid}")
            stop_vllm_process(process, wait_timeout=5)
            killed_count += 1
        else:
            _VLLM_PROCESSES.remove(process)
    logger.info(f"Killed {killed_count} VLLM processes")
    return killed_count


def _is_server_running(port: int) -> bool:
    """Check if server is running on port."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_base_client(
    client=None, cache: bool = True, api_key="abc", vllm_cmd=None, vllm_process=None
) -> OpenAI:
    """Get OpenAI client from various inputs."""
    from llm_utils import MOpenAI

    if client is None:
        if vllm_cmd is not None:
            # Parse environment variables from command to get clean command for port extraction
            _, cleaned_cmd = _parse_env_vars_from_cmd(vllm_cmd)
            port = _extract_port_from_vllm_cmd(cleaned_cmd)
            return MOpenAI(
                base_url=f"http://localhost:{port}/v1", api_key=api_key, cache=cache
            )
        raise ValueError("Either client or vllm_cmd must be provided.")
    if isinstance(client, int):
        return MOpenAI(
            base_url=f"http://localhost:{client}/v1", api_key=api_key, cache=cache
        )
    if isinstance(client, str):
        return MOpenAI(base_url=client, api_key=api_key, cache=cache)
    if isinstance(client, OpenAI):
        return MOpenAI(base_url=client.base_url, api_key=api_key, cache=cache)
    raise ValueError(
        "Invalid client type. Must be OpenAI, port (int), base_url (str), or None."
    )


def _is_lora_path(path: str) -> bool:
    """Check if path is LoRA adapter directory."""
    if not os.path.isdir(path):
        return False
    adapter_config_path = os.path.join(path, "adapter_config.json")
    return os.path.isfile(adapter_config_path)


def _get_port_from_client(client: OpenAI) -> int | None:
    """Extract port from OpenAI client base_url."""
    if hasattr(client, "base_url") and client.base_url:
        base_url = str(client.base_url)
        if "localhost:" in base_url:
            try:
                port_part = base_url.split("localhost:")[1].split("/")[0]
                return int(port_part)
            except (IndexError, ValueError):
                pass
    return None


def _load_lora_adapter(lora_path: str, port: int) -> str:
    """Load LoRA adapter from path."""
    lora_name = os.path.basename(lora_path.rstrip("/\\"))
    if not lora_name:
        lora_name = os.path.basename(os.path.dirname(lora_path))

    response = requests.post(
        f"http://localhost:{port}/v1/load_lora_adapter",
        headers={"accept": "application/json", "Content-Type": "application/json"},
        json={"lora_name": lora_name, "lora_path": os.path.abspath(lora_path)},
    )
    response.raise_for_status()
    return lora_name


def _unload_lora_adapter(lora_path: str, port: int) -> None:
    """Unload LoRA adapter."""
    try:
        lora_name = os.path.basename(lora_path.rstrip("/\\"))
        if not lora_name:
            lora_name = os.path.basename(os.path.dirname(lora_path))

        response = requests.post(
            f"http://localhost:{port}/v1/unload_lora_adapter",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            json={"lora_name": lora_name, "lora_int_id": 0},
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Error unloading LoRA adapter: {str(e)[:100]}")
