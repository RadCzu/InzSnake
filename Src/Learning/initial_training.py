from Src.Learning.TrainingProcess import AITrainingProcess

save_path = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\Models\\FoodEnd"
file_name = "initial_4"
snapshot_name = "initial_snapshot_4"
score_file = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\LearningHistory\\FoodEnd\\box_food_end_3.csv"
tester = AITrainingProcess(save_path, file_name, snapshot_name, score_file)
#
# tester.movement_training(num_episodes=1000)
#
tester.train_single(num_episodes=10000)

tester.example_game_single()
tester.example_game_single()


