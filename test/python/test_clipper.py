import numpy as np
from clipperpluspy import find_clique, ClipperParams
import unittest

class TestClipperplus(unittest.TestCase):
    
    test_data = [
        {"name": "Adjacency Matrix 1",
         "adj": np.array([[0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                          [0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                          [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                          [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                          [1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                          [0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                          [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                          [0, 1, 1, 1, 1, 1, 0, 1, 1, 0]]),
         "expected_clique_size": 6,
         "expected_clique": [1, 3, 4, 6, 7, 8],
         "expected_certificate": 0},
        {"name": "Adjacency Matrix 2",
         "adj": np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                          [1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                          [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
                          [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                          [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]]),
         "expected_clique_size": 7,
         "expected_clique": [6, 0, 2, 3, 5, 8, 9],
         "expected_certificate": 3},
        {"name": "Adjacency Matrix 3",
         "adj": np.array([[0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                          [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                          [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                          [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                          [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                          [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                          [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                          [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                          [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                          [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                          [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                          [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]]),
         "expected_clique_size": 8,
         "expected_clique": [4, 10, 13, 14, 15, 16, 17, 18],
         "expected_certificate": 0}
    ]
    
    def test_clique(self):
        # Set up parameters
        params = ClipperParams()
        params.check_lovasz_theta = False
        params.cuhallar_params.options = "/workspace/parameters/cuhallar_params_inexact.cfg"
        
        for i in range(len(self.test_data)):
            with self.subTest("Finding cliques on predetermined adjacency matrices", i=i):
                adj = self.test_data[i]["adj"]
                clique_size, clique, certificate = find_clique(adj, params)
                print(f"\nTest {i}: {self.test_data[i]['name']}\n")
                print(f"{self.test_data[i]['adj']}\n", flush=True)
                self.assertEqual(clique_size, self.test_data[i]["expected_clique_size"])
                # self.assertEqual(clique, self.test_data[i]["expected_clique"])
                # self.assertEqual(certificate, self.test_data[i]["expected_certificate"])
                print(flush=True)
                
if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)