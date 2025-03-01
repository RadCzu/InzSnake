from Src.Learning.TrainingProcess import AITrainingProcess

save_path = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\Models\\FoodEnd"
file_name = "clear_test_1"
snapshot_name = "clear_brain"
score_file = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\LearningHistory\\FoodEnd\\new_score.csv"
tester = AITrainingProcess(save_path, file_name, snapshot_name, score_file)

tester.train(num_episodes=10000, game_length=200)

tester.example_game(200)
tester.example_game(200)

