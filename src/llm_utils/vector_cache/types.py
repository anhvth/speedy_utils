"""Type definitions for the embed_cache package."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# Type aliases
TextList = list[str]
EmbeddingArray = NDArray[np.float32]
EmbeddingList = list[list[float]]
CacheStats = dict[str, int]
ModelIdentifier = str  # Either URL or model name/path

# For backwards compatibility
Embeddings = Union[EmbeddingArray, EmbeddingList]
