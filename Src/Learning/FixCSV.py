import csv
def fix_csv(file_path):
    # Read the existing data
    with open(file_path, mode='r', newline='') as file:
        reader = list(csv.reader(file))

    # Ensure the file is not empty and has a header
    if not reader or len(reader) < 2:
        print("CSV file is empty or only contains a header!")
        return

    # Fix the first column (Episode numbers)
    fixed_data = [reader[0]]  # Keep the header
    for i, row in enumerate(reader[1:]):  # Skip the header row
        row[0] = str(i * 10)  # Set the first column value (0, 10, 20, ...)
        fixed_data.append(row)

    # Write the corrected data back to the file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(fixed_data)

    print("CSV file has been successfully fixed.")

import matplotlib.pyplot as plt
import numpy as np

def plot_csv(file_path, save_path="training_progress.png"):
    # Read the data
    episodes = []
    scores = []

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            episodes.append(int(row[0]))  # Episode numbers
            scores.append(float(row[1]))  # Scores

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, marker='o', linestyle='-', markersize=3, label="Przeciętny wynik")

    # Limit the number of x-axis ticks to ~10
    num_ticks = 10
    indices = np.linspace(0, len(episodes) - 1, num_ticks, dtype=int)
    plt.xticks([episodes[i] for i in indices], rotation=45)  # Set spaced x-axis labels

    # Labels and title
    plt.xlabel("Epizod")
    plt.ylabel("Przeciętny wynik")
    plt.title("Proces nauczania Agenta")
    plt.legend()
    plt.grid()

    # Save the plot as an image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High resolution (300 DPI)

    # Show the plot
    plt.show()


plot_csv("C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\LearningHistory\\FoodEnd\\multiplayer.csv", "nauczanie_etap_3.png")