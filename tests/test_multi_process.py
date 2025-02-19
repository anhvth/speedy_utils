import unittest
from speedy_utils.multi_worker.process import multi_process

class TestMultiProcess(unittest.TestCase):
    def test_add_one(self):
        # Define a simple function to increment input 'x' by 1.
        def add_one(x: int) -> int:
            return x + 1

        # Create a sequence of inputs: list of dictionaries.
        inputs = list(range(5))
        # Execute multi_process with 2 workers and silent progress.
        results = multi_process(add_one, inputs, workers=2, verbose=False)
        # Assert that each element is incremented correctly.
        self.assertEqual(results, [i + 1 for i in range(5)])

if __name__ == '__main__':
    unittest.main()
