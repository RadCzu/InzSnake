from Src.Learning.TrainingProcess import AITrainingProcess

save_path = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\Models\\FoodEnd"
file_name = "foodless_network_2"
snapshot_name = "new_snake_brain_v3"
score_file = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\LearningHistory\\FoodEnd\\foodless_1.csv"
tester = AITrainingProcess(save_path, file_name, snapshot_name, score_file)

tester.train_single(num_episodes=8500)

tester.example_game_single()
tester.example_game_single()


