import unittest

from qmodel import compile_model, topology


class TestModel(unittest.TestCase):

    def test_model(self):
        # Prepare model A
        #model = compile_model()

        # Override default config
        model = compile_model(layers=topology({
            "hidden_units": 2048
        }))

        print(model)

        # Load Weights
        # model.load_weights("....")


if __name__ == '__main__':
    unittest.main()
