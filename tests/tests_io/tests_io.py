from pyae.io import make_signal_ids
import unittest

# Unit test for the function
class TestMakeTargetLoad(unittest.TestCase):
    def test_make_target_load(self):
        # Create sample input data
        data = np.zeros((2, 3, 4, 5))  # shape (2 samples, 3 loads, 4 sensors, 5 steps)
        
        # Compute expected output
        n_samples, n_loads, n_sensors, n_steps = data.shape
        n_examples = n_samples * n_sensors
        expected_output = torch.tensor(
            np.repeat([x for x in range(n_loads)], repeats=n_examples), 
            dtype=torch.int32
        ).squeeze()
        
        # Call the function
        result = make_target_load(data)
        
        # Assert the output
        self.assertTrue(torch.equal(result, expected_output))

        def test_make_signal_ids(self):
            # Create sample input data
            data = np.zeros((2, 3, 4, 5))  # shape (2 samples, 3 loads, 4 sensors, 5 steps)
            
            # Compute expected output
            n_samples, n_loads, n_sensors, n_steps = data.shape
            n_examples = np.prod([n_samples])
            
            load_vector = np.arange(n_loads)
            sensor_vector = np.arange(n_sensors)
            
            expected_combinations = n_examples * list(product(load_vector, sensor_vector))
            expected_output = pd.MultiIndex.from_tuples(expected_combinations, names=["load", "sensor"])
            
            # Call the function
            result = make_signal_ids(data)
            
            # Assert the output
            self.assertTrue(result.equals(expected_output))

if __name__ == '__main__':
    unittest.main()