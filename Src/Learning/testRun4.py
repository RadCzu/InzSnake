from Src.Learning.TrainingProcess import AITrainingProcess

save_path = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\Models\\FoodEnd"
file_name = "empty"
snapshot_name = "empty"
score_file = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\LearningHistory\\FoodEnd\\mo.csv"
tester = AITrainingProcess(save_path, file_name, snapshot_name, score_file)

tester.example_game(0)
tester.example_game(0)

