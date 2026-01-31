# Dataset Processing Configuration Reference

This document provides configuration templates for common dataset processing scenarios.

## Configuration 1: Standard Tokenization (GPT2 Tokenizer)

```python
python example_tokenize_pack.py \
    --src /path/to/raw/dataset \
    --dst /path/to/tokenized/dataset \
    --tokenizer gpt2 \
    --seq_len 2048 \
    --workers 4 \
    --backend mp
```

**Use for:** Simple text tokenization for LLM fine-tuning

**Key parameters:**
- `seq_len=2048`: Standard context length
- `workers=4`: Adjust based on CPU cores
- `backend=mp`: Multiprocessing for CPU-bound work

---

## Configuration 2: Large-Scale Dataset (Ray Distributed)

```python
python example_tokenize_pack.py \
    --src /path/to/huge/dataset \
    --dst /path/to/output/dataset \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --seq_len 4096 \
    --workers 16 \
    --backend ray
```

**Use for:** 10M+ row datasets across multiple machines with Ray

**Key parameters:**
- `seq_len=4096`: Larger context for modern models
- `workers=16`: Can exceed local CPU count with Ray
- `backend=ray`: Distributed computing across cluster

---

## Configuration 3: Debug Mode (Small Test Run)

```python
python example_tokenize_pack.py \
    --src /path/to/dataset \
    --dst /path/to/output/test \
    --tokenizer gpt2 \
    --seq_len 2048 \
    --workers 1 \
    --debug
```

**Use for:** Testing the pipeline before full run

**Key parameters:**
- `--debug`: Process only first 1000 rows
- `workers=1`: Single worker for easier debugging
- No output size limit for validation

---

## Configuration 4: Memory-Constrained Environment

```python
python example_tokenize_pack.py \
    --src /path/to/dataset \
    --dst /path/to/output \
    --tokenizer gpt2 \
    --seq_len 512 \
    --workers 2 \
    --backend mp
```

**Use for:** Machines with limited RAM

**Key parameters:**
- `seq_len=512`: Smaller sequences use less memory per worker
- `workers=2`: Fewer workers = less memory overhead
- Each worker uses ~1-2GB typically

---

## Configuration 5: Fast Processing (High Throughput)

```python
python example_tokenize_pack.py \
    --src /path/to/dataset \
    --dst /path/to/output \
    --tokenizer gpt2 \
    --seq_len 2048 \
    --workers $(nproc) \
    --backend mp
```

**Use for:** Maximum throughput, CPU cores fully saturated

**Key parameters:**
- `workers=$(nproc)`: Match exact CPU count
- `backend=mp`: Multiprocessing for CPU optimization
- Expect 1000-10000 sequences/sec depending on tokenizer

---

## Worker Count Guidelines

| Scenario | Workers | Backend |
|----------|---------|---------|
| **Laptop (4 cores)** | 2-3 | mp |
| **Workstation (16 cores)** | 12-15 | mp |
| **Server (32 cores)** | 28-31 | mp |
| **Large cluster** | 100+ | ray |

**Rule of thumb:** `workers = cpu_count - 1` for good performance

---

## Sequence Length Recommendations

| Use Case | Seq Length | Memory/Worker |
|----------|------------|---------------|
| **Short text (summaries)** | 512 | ~400MB |
| **Medium text (articles)** | 2048 | ~800MB |
| **Long context (code)** | 4096 | ~1.5GB |
| **Very long (books)** | 8192 | ~3GB |

---

## Performance Tuning

### If you run out of memory:
1. Reduce `seq_len` by 50%
2. Reduce `workers` by 50%
3. Add swap space (fallback, slower)

### If processing is too slow:
1. Increase `workers` (up to CPU count)
2. Check if I/O is bottleneck (`--debug` for quick check)
3. Use `--backend ray` for distributed processing

### If some shards fail:
1. Check temp directory for partial files
2. Reduce `workers` (less memory pressure)
3. Add error logging (already implemented)

---

## Platform-Specific Notes

### Linux/Mac
```bash
# Full parallelism
--workers $(nproc) --backend mp

# With Ray cluster
--workers 64 --backend ray
```

### Windows
```bash
# Use fewer workers (GIL limitations)
--workers 2-4 --backend mp

# Consider WSL2 + Linux for better parallelism
```

### Docker/Container
```bash
# Respect container limits
--workers 2-4  # Even if host has more cores

# Set memory limits safely
--seq_len 1024  # Conservative
```

---

## Monitoring Metrics

Track these during processing:

```
Time per shard: <time_per_shard>
Tokens/second: len(output) * seq_len / total_time
Memory/worker: Watch via `top` or Docker stats
Failed shards: Count of None results
```

**Healthy run:**
- ✅ Tokens/sec > 1000
- ✅ Memory < 80% of available
- ✅ Failed shards = 0
- ✅ Total time = dataset_size / throughput

**Concerning run:**
- ⚠️ Tokens/sec < 100
- ⚠️ Memory > 90%
- ⚠️ Failed shards > 0
- ⚠️ Very uneven shard times

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `OOM Killed` | Too large seq_len or workers | Reduce both by 50% |
| `Pickle error` | Large object in args | Pass paths, not objects |
| `Timeout` | Shard too large | Increase workers or reduce seq_len |
| `Permission denied` | Temp directory | Use `sudo rm -rf ...` or `--debug` |
| `Empty result` | All examples filtered | Check transform logic |

