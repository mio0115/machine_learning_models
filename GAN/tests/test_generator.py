import unittest

from models.model_def import Generator


class TestGenerator(unittest.TestCase):

    def test_build_generator(self):
        config = {
            "origin_output_shape":(32, 32, 3),
            "num_of_conv":5,
            "filters":[32, 64, 128, 64, 3],
            "kernel_size":[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
            "strides":[1, 1, 1, 1, 1]
        }
        Generator(**config)


if __name__ == "__main__":
    print("Import Success")
