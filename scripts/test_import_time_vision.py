import time


t = time.time()
from vision_utils import *


load_time = time.time() - t
print(f'Imported vision_utils in {load_time:.4f} seconds')
