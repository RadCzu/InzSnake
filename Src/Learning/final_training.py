from Src.Learning.TrainingProcess import AITrainingProcess

save_path = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\Models\\FoodEnd"
file_name = "foodless_network_3"
snapshot_name = "new_snake_brain_v3"
score_file = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\LearningHistory\\FoodEnd\\multiplayer.csv"
tester = AITrainingProcess(save_path, file_name, snapshot_name, score_file)

tester.train(num_episodes=12300)

tester.example_game()
tester.example_game()

