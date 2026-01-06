import os

from speedy_utils import dump_json_or_pickle, fast_load_jsonl, load_by_ext


def main():
    # 1. Create some dummy data
    data = [{"id": i, "value": f"item_{i}"} for i in range(100)]

    # 2. Dump to JSONL
    print("Dumping to data.jsonl...")
    dump_json_or_pickle(data, 'data.jsonl')

    # 3. Dump to Pickle
    print("Dumping to data.pkl...")
    dump_json_or_pickle(data, 'data.pkl')

    # 4. Load using load_by_ext
    print("Loading data.pkl...")
    loaded_pkl = load_by_ext('data.pkl')
    print(f"Loaded {len(loaded_pkl)} items from pickle.")

    # 5. Stream using fast_load_jsonl
    print("Streaming data.jsonl...")
    count = 0
    for item in fast_load_jsonl('data.jsonl', progress=True):
        count += 1
    print(f"Streamed {count} items from jsonl.")

    # Cleanup
    os.remove('data.jsonl')
    os.remove('data.pkl')

if __name__ == "__main__":
    main()
