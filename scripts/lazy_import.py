from typing import TYPE_CHECKING

import lazy_loader as lazy


torch = lazy.load('torch')
if TYPE_CHECKING:
    import torch

ran = torch.random.rand(2, 2)
# print(ran)
