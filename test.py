import unittest
from speedy_utils import (
    SPEED_CACHE_DIR, ICACHE, mkdir_or_exist, dump_jsonl, dump_json_or_pickle, timef,
    load_json_or_pickle, load_by_ext, identify, memoize, imemoize, imemoize_v2,
    flatten_list, fprint, get_arg_names, memoize_v2, is_interactive, print_table,
    convert_to_builtin_python, Clock, multi_thread, multi_process, async_multi_thread
)

class TestSpeedyInit(unittest.TestCase):

    def test_SPEED_CACHE_DIR(self):
        self.assertIsNotNone(SPEED_CACHE_DIR)

    def test_ICACHE(self):
        self.assertIsNotNone(ICACHE)

    def test_mkdir_or_exist(self):
        self.assertTrue(callable(mkdir_or_exist))

    def test_dump_jsonl(self):
        self.assertTrue(callable(dump_jsonl))

    def test_dump_json_or_pickle(self):
        self.assertTrue(callable(dump_json_or_pickle))

    def test_timef(self):
        self.assertTrue(callable(timef))

    def test_load_json_or_pickle(self):
        self.assertTrue(callable(load_json_or_pickle))

    def test_load_by_ext(self):
        self.assertTrue(callable(load_by_ext))

    def test_identify(self):
        self.assertTrue(callable(identify))

    def test_memoize(self):
        self.assertTrue(callable(memoize))

    def test_imemoize(self):
        self.assertTrue(callable(imemoize))

    def test_imemoize_v2(self):
        self.assertTrue(callable(imemoize_v2))

    def test_flatten_list(self):
        self.assertTrue(callable(flatten_list))

    def test_fprint(self):
        self.assertTrue(callable(fprint))

    def test_get_arg_names(self):
        self.assertTrue(callable(get_arg_names))

    def test_memoize_v2(self):
        self.assertTrue(callable(memoize_v2))

    def test_is_interactive(self):
        self.assertTrue(callable(is_interactive))

    def test_print_table(self):
        self.assertTrue(callable(print_table))

    def test_convert_to_builtin_python(self):
        self.assertTrue(callable(convert_to_builtin_python))

    def test_Clock(self):
        self.assertTrue(callable(Clock))

    def test_multi_thread(self):
        self.assertTrue(callable(multi_thread))

    def test_multi_process(self):
        self.assertTrue(callable(multi_process))

    def test_async_multi_thread(self):
        self.assertTrue(callable(async_multi_thread))

if __name__ == '__main__':
    unittest.main()
