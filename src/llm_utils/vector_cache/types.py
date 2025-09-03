"""Type definitions for the embed_cache package."""

from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

# Type aliases
TextList = List[str]
EmbeddingArray = NDArray[np.float32]
EmbeddingList = List[List[float]]
CacheStats = Dict[str, int]
ModelIdentifier = str  # Either URL or model name/path

# For backwards compatibility
Embeddings = Union[EmbeddingArray, EmbeddingList]