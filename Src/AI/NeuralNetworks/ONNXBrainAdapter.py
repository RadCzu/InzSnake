import onnxruntime as ort
import numpy as np

class ONNXBrainAdapter:
    def __init__(self, model_path):
        # Load the ONNX model using onnxruntime
        self.session = ort.InferenceSession(model_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

    def run(self, map_summary, local_map, distance_from_walls, food_info, move_history, move_direction, device='cpu'):
        # Create a dictionary for the inputs in the same order as the PyTorch class
        inputs = {
            "map_summary": map_summary,
            "local_map": local_map,
            "distance_from_walls": distance_from_walls,
            "food_info": food_info,
            "move_history": move_history,
            "move_direction": move_direction
        }

        # Ensure the input names from the model match the provided input data
        for input_name in inputs:
            if input_name not in self.input_names:
                raise ValueError(f"Input name {input_name} is not in the model.")

        # Prepare the data (convert to numpy arrays as onnxruntime expects numpy arrays)
        input_data = {name: np.array(data, dtype=np.float32) for name, data in inputs.items()}

        # Run the model
        output = self.session.run(self.output_names, input_data)

        # Return the first output (adjust this if your model has multiple outputs)
        return output[0]

    def dispose(self):
        self.session = None

# Example usage:
if __name__ == "__main__":
    # Load the model
    model_path = 'path_to_your_model.onnx'
    snake_brain = SnakeBrainAdapter(model_path)

    # Create dummy input data (you would replace this with your actual data)
    map_summary = np.random.rand(1, 3, 64, 64).astype(np.float32)
    local_map = np.random.rand(1, 3, 15, 15).astype(np.float32)
    distance_from_walls = np.random.rand(1, 4).astype(np.float32)
    food_info = np.random.rand(1, 5 * 3 + 4).astype(np.float32)
    move_history = np.random.rand(1, 20 * 6 + 4).astype(np.float32)
    move_direction = np.random.rand(1, 4).astype(np.float32)

    # Run the model
    output = snake_brain.run(map_summary, local_map, distance_from_walls, food_info, move_history, move_direction)

    print("Model Output:", output)
