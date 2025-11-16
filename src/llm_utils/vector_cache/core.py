from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from time import time
from typing import Any, Dict, Literal, Optional, cast

import numpy as np


class VectorCache:
    """
    A caching layer for text embeddings with support for multiple backends.

    This cache is designed to be safe for multi-process environments where multiple
    processes may access the same cache file simultaneously. It uses SQLite WAL mode
    and retry logic with exponential backoff to handle concurrent access.

    Examples:
        # OpenAI API
        from llm_utils import VectorCache
        cache = VectorCache("https://api.openai.com/v1", api_key="your-key")
        embeddings = cache.embeds(["Hello world", "How are you?"])

        # Custom OpenAI-compatible server (auto-detects model)
        cache = VectorCache("http://localhost:8000/v1", api_key="abc")

        # Transformers (Sentence Transformers)
        cache = VectorCache("sentence-transformers/all-MiniLM-L6-v2")

        # vLLM (local model)
        cache = VectorCache("/path/to/model")

        # Explicit backend specification
        cache = VectorCache("model-name", backend="transformers")

        # Eager loading (default: False) - load model immediately for better performance
        cache = VectorCache("model-name", lazy=False)

        # Lazy loading - load model only when needed (may cause performance issues)
        cache = VectorCache("model-name", lazy=True)

    Multi-Process Safety:
        The cache uses SQLite WAL (Write-Ahead Logging) mode and implements retry logic
        with exponential backoff to handle database locks. Multiple processes can safely
        read and write to the same cache file simultaneously.

        Race Condition Protection:
        - Uses INSERT OR IGNORE to prevent overwrites when multiple processes compute the same text
        - The first process to successfully cache a text wins, subsequent attempts are ignored
        - This ensures deterministic results even with non-deterministic embedding models

        For best performance in multi-process scenarios, consider:
        - Using separate cache files per process if cache hits are low
        - Coordinating cache warm-up to avoid redundant computation
        - Monitor for excessive lock contention in high-concurrency scenarios
    """

    def __init__(
        self,
        url_or_model: str,
        backend: Literal["vllm", "transformers", "openai"] | None = None,
        embed_size: int | None = None,
        db_path: str | None = None,
        # OpenAI API parameters
        api_key: str | None = "abc",
        model_name: str | None = None,
        # vLLM parameters
        vllm_gpu_memory_utilization: float = 0.5,
        vllm_tensor_parallel_size: int = 1,
        vllm_dtype: str = "auto",
        vllm_trust_remote_code: bool = False,
        vllm_max_model_len: int | None = None,
        # Transformers parameters
        transformers_device: str = "auto",
        transformers_batch_size: int = 32,
        transformers_normalize_embeddings: bool = True,
        transformers_trust_remote_code: bool = False,
        # SQLite parameters
        sqlite_chunk_size: int = 999,
        sqlite_cache_size: int = 10000,
        sqlite_mmap_size: int = 268435456,  # 256MB
        # Processing parameters
        embedding_batch_size: int = 20_000,
        # Other parameters
        verbose: bool = True,
        lazy: bool = False,
    ) -> None:
        self.url_or_model = url_or_model
        self.embed_size = embed_size
        self.verbose = verbose
        self.lazy = lazy

        self.backend = self._determine_backend(backend)
        if self.verbose and backend is None:
            print(f"Auto-detected backend: {self.backend}")

        # Store all configuration parameters
        self.config = {
            # OpenAI
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
            "model_name": self._try_infer_model_name(model_name),
            # vLLM
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": vllm_tensor_parallel_size,
            "vllm_dtype": vllm_dtype,
            "vllm_trust_remote_code": vllm_trust_remote_code,
            "vllm_max_model_len": vllm_max_model_len,
            # Transformers
            "transformers_device": transformers_device,
            "transformers_batch_size": transformers_batch_size,
            "transformers_normalize_embeddings": transformers_normalize_embeddings,
            "transformers_trust_remote_code": transformers_trust_remote_code,
            # SQLite
            "sqlite_chunk_size": sqlite_chunk_size,
            "sqlite_cache_size": sqlite_cache_size,
            "sqlite_mmap_size": sqlite_mmap_size,
            # Processing
            "embedding_batch_size": embedding_batch_size,
        }

        # Auto-detect model_name for OpenAI if using custom URL and default model
        if (
            self.backend == "openai"
            and model_name == "text-embedding-3-small"
            and self.url_or_model != "https://api.openai.com/v1"
        ):
            if self.verbose:
                print(f"Attempting to auto-detect model from {self.url_or_model}...")
            try:
                import openai

                client = openai.OpenAI(
                    base_url=self.url_or_model, api_key=self.config["api_key"]
                )
                models = client.models.list()
                if models.data:
                    detected_model = models.data[0].id
                    self.config["model_name"] = detected_model
                    model_name = detected_model  # Update for db_path computation
                    if self.verbose:
                        print(f"Auto-detected model: {detected_model}")
                else:
                    if self.verbose:
                        print("No models found, using default model")
            except Exception as e:
                if self.verbose:
                    print(f"Model auto-detection failed: {e}, using default model")
                # Fallback to default if auto-detection fails

        # Set default db_path if not provided
        if db_path is None:
            if self.backend == "openai":
                model_id = self.config["model_name"] or "openai-default"
            else:
                model_id = self.url_or_model
            safe_name = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:16]
            self.db_path = (
                Path.home() / ".cache" / "embed" / f"{self.backend}_{safe_name}.sqlite"
            )
        else:
            self.db_path = Path(db_path)

        # Ensure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self._optimize_connection()
        self._ensure_schema()
        self._model = None  # Lazy loading
        self._client = None  # For OpenAI client

        # Load model/client if not lazy
        if not self.lazy:
            if self.verbose:
                print(f"Loading {self.backend} model/client: {self.url_or_model}")
            if self.backend == "openai":
                self._load_openai_client()
            elif self.backend in ["vllm", "transformers"]:
                self._load_model()
            if self.verbose:
                print(f"✓ {self.backend.upper()} model/client loaded successfully")

    def _determine_backend(
        self, backend: Literal["vllm", "transformers", "openai"] | None
    ) -> str:
        """Determine the appropriate backend based on url_or_model and user preference."""
        if backend is not None:
            valid_backends = ["vllm", "transformers", "openai"]
            if backend not in valid_backends:
                raise ValueError(
                    f"Invalid backend '{backend}'. Must be one of: {valid_backends}"
                )
            return backend

        if self.url_or_model.startswith("http"):
            return "openai"

        # Default to vllm for local models
        return "vllm"

    def _try_infer_model_name(self, model_name: str | None) -> str | None:
        """Infer model name for OpenAI backend if not explicitly provided."""
        if model_name:
            return model_name
        if "https://" in self.url_or_model:
            model_name = "text-embedding-3-small"

        if "http://localhost" in self.url_or_model:
            from openai import OpenAI

            client = OpenAI(base_url=self.url_or_model, api_key="abc")
            model_name = client.models.list().data[0].id

        # Default model name
        print("Infer model name:", model_name)
        return model_name

    def _optimize_connection(self) -> None:
        """Optimize SQLite connection for bulk operations and multi-process safety."""
        # Performance optimizations for bulk operations
        self.conn.execute(
            "PRAGMA journal_mode=WAL"
        )  # Write-Ahead Logging for better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still safe
        self.conn.execute(
            f"PRAGMA cache_size={self.config['sqlite_cache_size']}"
        )  # Configurable cache
        self.conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp storage
        self.conn.execute(
            f"PRAGMA mmap_size={self.config['sqlite_mmap_size']}"
        )  # Configurable memory mapping

        # Multi-process safety improvements
        self.conn.execute(
            "PRAGMA busy_timeout=30000"
        )  # Wait up to 30 seconds for locks
        self.conn.execute(
            "PRAGMA wal_autocheckpoint=1000"
        )  # Checkpoint WAL every 1000 pages

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS cache (
            hash TEXT PRIMARY KEY,
            text TEXT,
            embedding BLOB
        )
        """
        )
        # Add index for faster lookups if it doesn't exist
        self.conn.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(hash)
        """
        )
        self.conn.commit()

    def _load_openai_client(self) -> None:
        """Load OpenAI client."""
        import openai

        self._client = openai.OpenAI(
            base_url=self.url_or_model, api_key=self.config["api_key"]
        )

    def _load_model(self) -> None:
        """Load the model for vLLM or Transformers."""
        if self.backend == "vllm":
            from vllm import LLM  # type: ignore[import-not-found]

            gpu_memory_utilization = cast(
                float, self.config["vllm_gpu_memory_utilization"]
            )
            tensor_parallel_size = cast(int, self.config["vllm_tensor_parallel_size"])
            dtype = cast(str, self.config["vllm_dtype"])
            trust_remote_code = cast(bool, self.config["vllm_trust_remote_code"])
            max_model_len = cast(int | None, self.config["vllm_max_model_len"])

            vllm_kwargs = {
                "model": self.url_or_model,
                "task": "embed",
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": tensor_parallel_size,
                "dtype": dtype,
                "trust_remote_code": trust_remote_code,
            }

            if max_model_len is not None:
                vllm_kwargs["max_model_len"] = max_model_len

            try:
                self._model = LLM(**vllm_kwargs)
            except (ValueError, AssertionError, RuntimeError) as e:
                error_msg = str(e).lower()
                if (
                    ("kv cache" in error_msg and "gpu_memory_utilization" in error_msg)
                    or (
                        "memory" in error_msg
                        and ("gpu" in error_msg or "insufficient" in error_msg)
                    )
                    or ("free memory" in error_msg and "initial" in error_msg)
                    or ("engine core initialization failed" in error_msg)
                ):
                    raise ValueError(
                        f"Insufficient GPU memory for vLLM model initialization. "
                        f"Current vllm_gpu_memory_utilization ({gpu_memory_utilization}) may be too low. "
                        f"Try one of the following:\n"
                        f"1. Increase vllm_gpu_memory_utilization (e.g., 0.5, 0.8, or 0.9)\n"
                        f"2. Decrease vllm_max_model_len (e.g., 4096, 8192)\n"
                        f"3. Use a smaller model\n"
                        f"4. Ensure no other processes are using GPU memory during initialization\n"
                        f"Original error: {e}"
                    ) from e
                raise
        elif self.backend == "transformers":
            import torch  # type: ignore[import-not-found] # noqa: F401
            from transformers import (  # type: ignore[import-not-found]
                AutoModel,
                AutoTokenizer,
            )

            device = self.config["transformers_device"]
            # Handle "auto" device selection - default to CPU for transformers to avoid memory conflicts
            if device == "auto":
                device = "cpu"  # Default to CPU to avoid GPU memory conflicts with vLLM

            tokenizer = AutoTokenizer.from_pretrained(
                self.url_or_model,
                padding_side="left",
                trust_remote_code=self.config["transformers_trust_remote_code"],
            )
            model = AutoModel.from_pretrained(
                self.url_or_model,
                trust_remote_code=self.config["transformers_trust_remote_code"],
            )

            # Move model to device
            model.to(device)
            model.eval()

            self._model = {"tokenizer": tokenizer, "model": model, "device": device}

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using the configured backend."""
        assert isinstance(texts, list), "texts must be a list"
        assert all(
            isinstance(t, str) for t in texts
        ), "all elements in texts must be strings"
        if self.backend == "openai":
            return self._get_openai_embeddings(texts)
        if self.backend == "vllm":
            return self._get_vllm_embeddings(texts)
        if self.backend == "transformers":
            return self._get_transformers_embeddings(texts)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _get_openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using OpenAI API."""
        assert isinstance(texts, list), "texts must be a list"
        assert all(
            isinstance(t, str) for t in texts
        ), "all elements in texts must be strings"
        # Assert valid model_name for OpenAI backend
        model_name = self.config["model_name"]
        assert (
            model_name is not None and model_name.strip()
        ), f"Invalid model_name for OpenAI backend: {model_name}. Model name must be provided and non-empty."

        if self._client is None:
            if self.verbose:
                print("🔧 Loading OpenAI client...")
            self._load_openai_client()
            if self.verbose:
                print("✓ OpenAI client loaded successfully")

        response = self._client.embeddings.create(  # type: ignore
            model=model_name, input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def _get_vllm_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using vLLM."""
        assert isinstance(texts, list), "texts must be a list"
        assert all(
            isinstance(t, str) for t in texts
        ), "all elements in texts must be strings"
        if self._model is None:
            if self.verbose:
                print("🔧 Loading vLLM model...")
            self._load_model()
            if self.verbose:
                print("✓ vLLM model loaded successfully")

        outputs = self._model.embed(texts)  # type: ignore
        embeddings = [o.outputs.embedding for o in outputs]
        return embeddings

    def _get_transformers_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using transformers directly."""
        assert isinstance(texts, list), "texts must be a list"
        assert all(
            isinstance(t, str) for t in texts
        ), "all elements in texts must be strings"
        if self._model is None:
            if self.verbose:
                print("🔧 Loading Transformers model...")
            self._load_model()
            if self.verbose:
                print("✓ Transformers model loaded successfully")

        if not isinstance(self._model, dict):
            raise ValueError("Model not loaded properly for transformers backend")

        tokenizer = self._model["tokenizer"]
        model = self._model["model"]
        device = self._model["device"]

        normalize_embeddings = cast(
            bool, self.config["transformers_normalize_embeddings"]
        )

        # For now, use a default max_length
        max_length = 8192

        # Tokenize
        batch_dict = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        # Run model
        import torch  # type: ignore[import-not-found]

        with torch.no_grad():
            outputs = model(**batch_dict)

        # Apply last token pooling
        embeddings = self._last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # Normalize if needed
        if normalize_embeddings:
            import torch.nn.functional as F  # type: ignore[import-not-found]

            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Apply last token pooling to get embeddings."""
        import torch  # type: ignore[import-not-found]

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _hash_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _execute_with_retry(self, query: str, params=None) -> sqlite3.Cursor:
        """Execute SQLite query with retry logic for multi-process safety."""
        max_retries = 3
        base_delay = 0.05  # 50ms base delay for reads (faster than writes)

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if params is None:
                    return self.conn.execute(query)
                return self.conn.execute(query, params)

            except sqlite3.OperationalError as e:
                last_exception = e
                if "database is locked" in str(e).lower() and attempt < max_retries:
                    # Exponential backoff: 0.05s, 0.1s, 0.2s
                    delay = base_delay * (2**attempt)
                    if self.verbose:
                        print(
                            f"⚠️  Database locked on read, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    import time

                    time.sleep(delay)
                    continue
                # Re-raise if not a lock error or max retries exceeded
                raise
            except Exception:
                # Re-raise any other exceptions
                raise

        # This should never be reached, but satisfy the type checker
        raise last_exception or RuntimeError("Failed to execute query after retries")

    def embeds(self, texts: list[str], cache: bool = True) -> np.ndarray:
        """
        Return embeddings for all texts.

        If cache=True, compute and cache missing embeddings.
        If cache=False, force recompute all embeddings and update cache.

        This method processes lookups and embedding generation in chunks to
        handle very large input lists. A tqdm progress bar is shown while
        computing missing embeddings.
        """
        assert isinstance(texts, list), "texts must be a list"
        assert all(
            isinstance(t, str) for t in texts
        ), "all elements in texts must be strings"
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        t = time()
        hashes = [self._hash_text(t) for t in texts]

        # Helper to yield chunks
        def _chunks(lst: list[str], n: int) -> list[list[str]]:
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        # Fetch known embeddings in bulk with optimized chunk size
        hit_map: dict[str, np.ndarray] = {}
        chunk_size = self.config["sqlite_chunk_size"]

        # Use bulk lookup with optimized query and retry logic
        hash_chunks = _chunks(hashes, chunk_size)
        for chunk in hash_chunks:
            placeholders = ",".join("?" * len(chunk))
            rows = self._execute_with_retry(
                f"SELECT hash, embedding FROM cache WHERE hash IN ({placeholders})",
                chunk,
            ).fetchall()
            for h, e in rows:
                hit_map[h] = np.frombuffer(e, dtype=np.float32)

        # Determine which texts are missing
        if cache:
            missing_items: list[tuple[str, str]] = [
                (t, h) for t, h in zip(texts, hashes, strict=False) if h not in hit_map
            ]
        else:
            missing_items: list[tuple[str, str]] = [
                (t, h) for t, h in zip(texts, hashes, strict=False)
            ]

        if missing_items:
            if self.verbose:
                print(
                    f"Computing {len(missing_items)}/{len(texts)} missing embeddings..."
                )
            self._process_missing_items_with_batches(missing_items, hit_map)

        # Return embeddings in the original order
        elapsed = time() - t
        if self.verbose:
            print(f"Retrieved {len(texts)} embeddings in {elapsed:.2f} seconds")
        return np.vstack([hit_map[h] for h in hashes])

    def _process_missing_items_with_batches(
        self, missing_items: list[tuple[str, str]], hit_map: dict[str, np.ndarray]
    ) -> None:
        """
        Process missing items in batches with simple progress tracking.
        """
        t = time()  # Track total processing time

        batch_size = self.config["embedding_batch_size"]
        total_items = len(missing_items)

        if self.verbose:
            print(
                f"Computing embeddings for {total_items} missing texts in batches of {batch_size}..."
            )
            if self.backend in ["vllm", "transformers"] and self._model is None:
                print("⚠️  Model will be loaded on first batch (lazy loading enabled)")
            elif self.backend in ["vllm", "transformers"]:
                print("✓ Model already loaded, ready for efficient batch processing")

        # Track total committed items
        total_committed = 0
        processed_count = 0

        # Process in batches
        for i in range(0, total_items, batch_size):
            batch_items = missing_items[i : i + batch_size]
            batch_texts = [text for text, _ in batch_items]

            # Get embeddings for this batch
            batch_embeds = self._get_embeddings(batch_texts)

            # Prepare batch data for immediate insert
            batch_data: list[tuple[str, str, bytes]] = []
            for (text, h), vec in zip(batch_items, batch_embeds, strict=False):
                arr = np.asarray(vec, dtype=np.float32)
                batch_data.append((h, text, arr.tobytes()))
                hit_map[h] = arr

            # Immediate commit after each batch
            self._bulk_insert(batch_data)
            total_committed += len(batch_data)

            # Update progress - simple single line
            batch_size_actual = len(batch_items)
            processed_count += batch_size_actual
            if self.verbose:
                elapsed = time() - t
                rate = processed_count / elapsed if elapsed > 0 else 0
                progress_pct = (processed_count / total_items) * 100
                print(
                    f"\rProgress: {processed_count}/{total_items} ({progress_pct:.1f}%) | {rate:.0f} texts/sec",
                    end="",
                    flush=True,
                )

        if self.verbose:
            total_time = time() - t
            rate = total_items / total_time if total_time > 0 else 0
            print(
                f"\n✅ Completed: {total_items} embeddings computed and {total_committed} items committed to database"
            )
            print(f"   Total time: {total_time:.2f}s | Rate: {rate:.1f} embeddings/sec")

    def __call__(self, texts: list[str], cache: bool = True) -> np.ndarray:
        assert isinstance(texts, list), "texts must be a list"
        assert all(
            isinstance(t, str) for t in texts
        ), "all elements in texts must be strings"
        return self.embeds(texts, cache)

    def _bulk_insert(self, data: list[tuple[str, str, bytes]]) -> None:
        """
        Perform bulk insert of embedding data with retry logic for multi-process safety.

        Uses INSERT OR IGNORE to prevent race conditions where multiple processes
        might try to insert the same text hash. The first process to successfully
        insert wins, subsequent attempts are ignored. This ensures deterministic
        caching behavior in multi-process environments.
        """
        if not data:
            return

        max_retries = 3
        base_delay = 0.1  # 100ms base delay

        for attempt in range(max_retries + 1):
            try:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO cache (hash, text, embedding) VALUES (?, ?, ?)",
                    data,
                )
                self.conn.commit()

                # Check if some insertions were ignored due to existing entries
                # if self.verbose and cursor.rowcount < len(data):
                # ignored_count = len(data) - cursor.rowcount
                # if ignored_count > 0:
                #     print(f"ℹ️  {ignored_count}/{len(data)} embeddings already existed in cache (computed by another process)")

                return  # Success, exit the retry loop

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s
                    delay = base_delay * (2**attempt)
                    if self.verbose:
                        print(
                            f"⚠️  Database locked, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    import time

                    time.sleep(delay)
                    continue
                # Re-raise if not a lock error or max retries exceeded
                raise
            except Exception:
                # Re-raise any other exceptions
                raise

    def get_cache_stats(self) -> dict[str, int]:
        """Get statistics about the cache."""
        cursor = self._execute_with_retry("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        return {"total_cached": count}

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        max_retries = 3
        base_delay = 0.1  # 100ms base delay

        for attempt in range(max_retries + 1):
            try:
                self.conn.execute("DELETE FROM cache")
                self.conn.commit()
                return  # Success

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    if self.verbose:
                        print(
                            f"⚠️  Database locked during clear, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    import time

                    time.sleep(delay)
                    continue
                raise
            except Exception:
                raise

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return {
            "url_or_model": self.url_or_model,
            "backend": self.backend,
            "embed_size": self.embed_size,
            "db_path": str(self.db_path),
            "verbose": self.verbose,
            "lazy": self.lazy,
            **self.config,
        }

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            elif key == "verbose":
                self.verbose = value
            elif key == "lazy":
                self.lazy = value
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Reset model if backend-specific parameters changed
        backend_params = {
            "vllm": [
                "vllm_gpu_memory_utilization",
                "vllm_tensor_parallel_size",
                "vllm_dtype",
                "vllm_trust_remote_code",
                "vllm_max_model_len",
            ],
            "transformers": [
                "transformers_device",
                "transformers_batch_size",
                "transformers_normalize_embeddings",
                "transformers_trust_remote_code",
            ],
            "openai": ["api_key", "model_name"],
            "processing": ["embedding_batch_size"],
        }

        if any(param in kwargs for param in backend_params.get(self.backend, [])):
            self._model = None  # Force reload on next use
            if self.backend == "openai":
                self._client = None

    def __del__(self) -> None:
        """Clean up database connection."""
        if hasattr(self, "conn"):
            self.conn.close()
