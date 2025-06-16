# VLLM Server Examples

This directory contains scripts for working with VLLM servers.

## Files

- `serve_script.sh` - Script to start the VLLM server
- `example_vllm_client.py` - Beautiful example client for interacting with VLLM
- `requirements_example.txt` - Python dependencies for the example

## Usage

### 1. Start the VLLM Server

```bash
bash serve_script.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements_example.txt
```

### 3. Run Examples

```bash
python example_vllm_client.py
```

## Features

The example client demonstrates:

- ✅ Basic text generation
- ✅ Batch processing
- ✅ Creative writing with high temperature
- ✅ Code generation with low temperature
- ✅ Proper error handling
- ✅ Health checks
- ✅ Beautiful logging with loguru
- ✅ Type safety with Pydantic models
- ✅ Async/await patterns

## Configuration

The client connects to `http://localhost:8140` by default.
Modify the `VLLMClient` initialization to use different servers.
