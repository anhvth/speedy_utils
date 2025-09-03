from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Literal, Optional, cast

import numpy as np


class VectorCache:
    """
    A caching layer for text embeddings with support for multiple backends.
    
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
        
        # Lazy loading (default: True) - load model only when needed
        cache = VectorCache("model-name", lazy=True)
        
        # Eager loading - load model immediately
        cache = VectorCache("model-name", lazy=False)
    """
    def __init__(
        self,
        url_or_model: str,
        backend: Optional[Literal["vllm", "transformers", "openai"]] = None,
        embed_size: Optional[int] = None,
        db_path: Optional[str] = None,
        # OpenAI API parameters
        api_key: Optional[str] = "abc",
        model_name: Optional[str] = None,
        # vLLM parameters
        vllm_gpu_memory_utilization: float = 0.5,
        vllm_tensor_parallel_size: int = 1,
        vllm_dtype: str = "auto",
        vllm_trust_remote_code: bool = False,
        vllm_max_model_len: Optional[int] = None,
        # Transformers parameters
        transformers_device: str = "auto",
        transformers_batch_size: int = 32,
        transformers_normalize_embeddings: bool = True,
        transformers_trust_remote_code: bool = False,
        # SQLite parameters
        sqlite_chunk_size: int = 999,
        sqlite_cache_size: int = 10000,
        sqlite_mmap_size: int = 268435456,
        # Other parameters
        verbose: bool = True,
        lazy: bool = True,
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
            "model_name": model_name,
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
        }
        
        # Auto-detect model_name for OpenAI if using custom URL and default model
        if (self.backend == "openai" and 
            model_name == "text-embedding-3-small" and 
            self.url_or_model != "https://api.openai.com/v1"):
            if self.verbose:
                print(f"Attempting to auto-detect model from {self.url_or_model}...")
            try:
                import openai
                client = openai.OpenAI(
                    base_url=self.url_or_model, 
                    api_key=self.config["api_key"]
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
                pass
        
        # Set default db_path if not provided
        if db_path is None:
            if self.backend == "openai":
                model_id = self.config["model_name"] or "openai-default"
            else:
                model_id = self.url_or_model
            safe_name = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:16]
            self.db_path = Path.home() / ".cache" / "embed" / f"{self.backend}_{safe_name}.sqlite"
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
            if self.backend == "openai":
                self._load_openai_client()
            elif self.backend in ["vllm", "transformers"]:
                self._load_model()

    def _determine_backend(self, backend: Optional[Literal["vllm", "transformers", "openai"]]) -> str:
        """Determine the appropriate backend based on url_or_model and user preference."""
        if backend is not None:
            valid_backends = ["vllm", "transformers", "openai"]
            if backend not in valid_backends:
                raise ValueError(f"Invalid backend '{backend}'. Must be one of: {valid_backends}")
            return backend
            
        if self.url_or_model.startswith("http"):
            return "openai"
        
        # Default to vllm for local models
        return "vllm"

    def _optimize_connection(self) -> None:
        """Optimize SQLite connection for bulk operations."""
        # Performance optimizations for bulk operations
        self.conn.execute(
            "PRAGMA journal_mode=WAL"
        )  # Write-Ahead Logging for better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still safe
        self.conn.execute(f"PRAGMA cache_size={self.config['sqlite_cache_size']}")  # Configurable cache
        self.conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp storage
        self.conn.execute(f"PRAGMA mmap_size={self.config['sqlite_mmap_size']}")  # Configurable memory mapping

    def _ensure_schema(self) -> None:
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            hash TEXT PRIMARY KEY,
            text TEXT,
            embedding BLOB
        )
        """)
        # Add index for faster lookups if it doesn't exist
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(hash)
        """)
        self.conn.commit()

    def _load_openai_client(self) -> None:
        """Load OpenAI client."""
        import openai
        self._client = openai.OpenAI(
            base_url=self.url_or_model, 
            api_key=self.config["api_key"]
        )

    def _load_model(self) -> None:
        """Load the model for vLLM or Transformers."""
        if self.backend == "vllm":
            from vllm import LLM
            
            gpu_memory_utilization = cast(float, self.config["vllm_gpu_memory_utilization"])
            tensor_parallel_size = cast(int, self.config["vllm_tensor_parallel_size"])
            dtype = cast(str, self.config["vllm_dtype"])
            trust_remote_code = cast(bool, self.config["vllm_trust_remote_code"])
            max_model_len = cast(Optional[int], self.config["vllm_max_model_len"])
            
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
                if ("kv cache" in error_msg and "gpu_memory_utilization" in error_msg) or \
                   ("memory" in error_msg and ("gpu" in error_msg or "insufficient" in error_msg)) or \
                   ("free memory" in error_msg and "initial" in error_msg) or \
                   ("engine core initialization failed" in error_msg):
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
                else:
                    raise
        elif self.backend == "transformers":
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            device = self.config["transformers_device"]
            # Handle "auto" device selection - default to CPU for transformers to avoid memory conflicts
            if device == "auto":
                device = "cpu"  # Default to CPU to avoid GPU memory conflicts with vLLM
            
            tokenizer = AutoTokenizer.from_pretrained(self.url_or_model, padding_side='left', trust_remote_code=self.config["transformers_trust_remote_code"])
            model = AutoModel.from_pretrained(self.url_or_model, trust_remote_code=self.config["transformers_trust_remote_code"])
            
            # Move model to device
            model.to(device)
            model.eval()
            
            self._model = {"tokenizer": tokenizer, "model": model, "device": device}

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using the configured backend."""
        if self.backend == "openai":
            return self._get_openai_embeddings(texts)
        elif self.backend == "vllm":
            return self._get_vllm_embeddings(texts)
        elif self.backend == "transformers":
            return self._get_transformers_embeddings(texts)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _get_openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using OpenAI API."""
        # Assert valid model_name for OpenAI backend
        model_name = self.config["model_name"]
        assert model_name is not None and model_name.strip(), f"Invalid model_name for OpenAI backend: {model_name}. Model name must be provided and non-empty."

        if self._client is None:
            self._load_openai_client()
        
        response = self._client.embeddings.create(  # type: ignore
            model=model_name, 
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def _get_vllm_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using vLLM."""
        if self._model is None:
            self._load_model()
        
        outputs = self._model.embed(texts)  # type: ignore
        embeddings = [o.outputs.embedding for o in outputs]
        return embeddings

    def _get_transformers_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using transformers directly."""
        if self._model is None:
            self._load_model()
        
        if not isinstance(self._model, dict):
            raise ValueError("Model not loaded properly for transformers backend")
        
        tokenizer = self._model["tokenizer"]
        model = self._model["model"]
        device = self._model["device"]
        
        normalize_embeddings = cast(bool, self.config["transformers_normalize_embeddings"])
        
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
        import torch
        with torch.no_grad():
            outputs = model(**batch_dict)
        
        # Apply last token pooling
        embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize if needed
        if normalize_embeddings:
            import torch.nn.functional as F
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Apply last token pooling to get embeddings."""
        import torch
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _hash_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def embeds(self, texts: list[str], cache: bool = True) -> np.ndarray:
        """
        Return embeddings for all texts.

        If cache=True, compute and cache missing embeddings.
        If cache=False, force recompute all embeddings and update cache.

        This method processes lookups and embedding generation in chunks to
        handle very large input lists. A tqdm progress bar is shown while
        computing missing embeddings.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        hashes = [self._hash_text(t) for t in texts]

        # Helper to yield chunks
        def _chunks(lst: list[str], n: int) -> list[list[str]]:
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        # Fetch known embeddings in bulk with optimized chunk size
        hit_map: dict[str, np.ndarray] = {}
        chunk_size = self.config["sqlite_chunk_size"]

        # Use bulk lookup with optimized query
        hash_chunks = _chunks(hashes, chunk_size)
        for chunk in hash_chunks:
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT hash, embedding FROM cache WHERE hash IN ({placeholders})",
                chunk,
            ).fetchall()
            for h, e in rows:
                hit_map[h] = np.frombuffer(e, dtype=np.float32)

        # Determine which texts are missing
        if cache:
            missing_items: list[tuple[str, str]] = [
                (t, h) for t, h in zip(texts, hashes) if h not in hit_map
            ]
        else:
            missing_items: list[tuple[str, str]] = [
                (t, h) for t, h in zip(texts, hashes)
            ]

        if missing_items:
            if self.verbose:
                print(f"Computing embeddings for {len(missing_items)} missing texts...")
            missing_texts = [t for t, _ in missing_items]
            embeds = self._get_embeddings(missing_texts)

            # Prepare batch data for bulk insert
            bulk_insert_data: list[tuple[str, str, bytes]] = []
            for (text, h), vec in zip(missing_items, embeds):
                arr = np.asarray(vec, dtype=np.float32)
                bulk_insert_data.append((h, text, arr.tobytes()))
                hit_map[h] = arr

            self._bulk_insert(bulk_insert_data)

        # Return embeddings in the original order
        return np.vstack([hit_map[h] for h in hashes])

    def __call__(self, texts: list[str], cache: bool = True) -> np.ndarray:
        return self.embeds(texts, cache)

    def _bulk_insert(self, data: list[tuple[str, str, bytes]]) -> None:
        """Perform bulk insert of embedding data."""
        if not data:
            return

        self.conn.executemany(
            "INSERT OR REPLACE INTO cache (hash, text, embedding) VALUES (?, ?, ?)",
            data,
        )
        self.conn.commit()

    def precompute_embeddings(self, texts: list[str]) -> None:
        """
        Precompute embeddings for a large list of texts efficiently.
        This is optimized for bulk operations when you know all texts upfront.
        """
        if not texts:
            return

        # Remove duplicates while preserving order
        unique_texts = list(dict.fromkeys(texts))
        if self.verbose:
            print(f"Precomputing embeddings for {len(unique_texts)} unique texts...")

        # Check which ones are already cached
        hashes = [self._hash_text(t) for t in unique_texts]
        existing_hashes = set()

        # Bulk check for existing embeddings
        chunk_size = self.config["sqlite_chunk_size"]
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT hash FROM cache WHERE hash IN ({placeholders})",
                chunk,
            ).fetchall()
            existing_hashes.update(h[0] for h in rows)

        # Find missing texts
        missing_items = [
            (t, h) for t, h in zip(unique_texts, hashes) if h not in existing_hashes
        ]

        if not missing_items:
            if self.verbose:
                print("All texts already cached!")
            return

        if self.verbose:
            print(f"Computing {len(missing_items)} missing embeddings...")
        missing_texts = [t for t, _ in missing_items]
        embeds = self._get_embeddings(missing_texts)

        # Prepare batch data for bulk insert
        bulk_insert_data: list[tuple[str, str, bytes]] = []
        for (text, h), vec in zip(missing_items, embeds):
            arr = np.asarray(vec, dtype=np.float32)
            bulk_insert_data.append((h, text, arr.tobytes()))

        self._bulk_insert(bulk_insert_data)
        if self.verbose:
            print(f"Successfully cached {len(missing_items)} new embeddings!")

    def get_cache_stats(self) -> dict[str, int]:
        """Get statistics about the cache."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        return {"total_cached": count}

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self.conn.execute("DELETE FROM cache")
        self.conn.commit()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "url_or_model": self.url_or_model,
            "backend": self.backend,
            "embed_size": self.embed_size,
            "db_path": str(self.db_path),
            "verbose": self.verbose,
            "lazy": self.lazy,
            **self.config
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
            "vllm": ["vllm_gpu_memory_utilization", "vllm_tensor_parallel_size", "vllm_dtype", 
                    "vllm_trust_remote_code", "vllm_max_model_len"],
            "transformers": ["transformers_device", "transformers_batch_size", 
                           "transformers_normalize_embeddings", "transformers_trust_remote_code"],
            "openai": ["api_key", "model_name"]
        }
        
        if any(param in kwargs for param in backend_params.get(self.backend, [])):
            self._model = None  # Force reload on next use
            if self.backend == "openai":
                self._client = None

    def __del__(self) -> None:
        """Clean up database connection."""
        if hasattr(self, "conn"):
            self.conn.close()
