import asyncio
import time

from speedy_utils import imemoize, memoize


# Sync Hybrid Cache
@memoize(cache_type='both')
def slow_square(x: int) -> int:
    print(f"Computing square of {x}...")
    time.sleep(1)
    return x * x

# Async Disk Cache
@memoize(cache_type='disk', cache_dir='./temp_cache')
async def async_slow_cube(x: int) -> int:
    print(f"Computing cube of {x}...")
    await asyncio.sleep(1)
    return x * x * x

# Interactive Cache
@imemoize
def interactive_op(x: int) -> int:
    print(f"Interactive op on {x}...")
    return x + 1

async def main():
    print("--- Sync Cache ---")
    print(slow_square(2))
    print(slow_square(2)) # Should be instant

    print("\n--- Async Cache ---")
    print(await async_slow_cube(3))
    print(await async_slow_cube(3)) # Should be instant

if __name__ == "__main__":
    asyncio.run(main())
