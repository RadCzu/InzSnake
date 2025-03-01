import torch
from torch import nn
from torch.autograd import Variable

# Assuming SnakeBrain is your model class
from Src.AI.NeuralNetworks.SnakeBrain import SnakeBrain  # Replace with your actual import

path = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\Models\\FoodEnd"
# Load the trained PyTorch model
model = SnakeBrain()
model.load_state_dict(torch.load(f"{path}\\foodless_network_2.pth"))
model.eval()  # Ensure the model is in evaluation mode

# Example input tensors to match your model's expected input shapes
dummy_map_summary = torch.randn(1, 3, 64, 64)  # Batch size 1, 4 channels, 64x64
dummy_local_map = torch.randn(1, 3, 15, 15)    # Batch size 1, 4 channels, 15x15
dummy_distance_from_walls = torch.randn(1, 4)  # Batch size 1, 4 features
dummy_food_info = torch.randn(1, 15)           # Batch size 1, 5*3 + 4
dummy_move_history = torch.randn(1, 120)       # Batch size 1, 20*6 + 4
dummy_move_direction = torch.randn(1, 4)       # Batch size 1, 4 features

# Create a tuple of inputs to match the forward method
dummy_inputs = (
    dummy_map_summary,
    dummy_local_map,
    dummy_distance_from_walls,
    dummy_food_info,
    dummy_move_history,
    dummy_move_direction,
)

# Export to ONNX
torch.onnx.export(
    model,                          # Model to export
    dummy_inputs,                   # Example inputs
    f"{path}\\snake_brain_retrained_2.onnx",             # Output ONNX file
    input_names=[
        "map_summary",
        "local_map",
        "distance_from_walls",
        "food_info",
        "move_history",
        "move_direction"
    ],
    output_names=["output"],        # Name of the output layer
    dynamic_axes={                  # Enable variable batch size for inputs
        "map_summary": {0: "batch_size"},
        "local_map": {0: "batch_size"},
        "distance_from_walls": {0: "batch_size"},
        "food_info": {0: "batch_size"},
        "move_history": {0: "batch_size"},
        "move_direction": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=11                # Specify the ONNX opset version
)

print("Model exported to ONNX!")