from Src.Learning.TrainingProcess import AITrainingProcess

save_path = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\Models\\FoodEnd"
file_name = "test1"
snapshot_name = "test1_snap"
score_file = "C:\\Users\\Radek\\Radek\\radek rc\\Uczelniane\\IV ROK\\InzSnake\\pythonProject\\Src\\Data\\LearningHistory\\FoodEnd"
tester = AITrainingProcess(save_path, file_name, snapshot_name, score_file)

tester.train(
    num_episodes=1000,
    game_length=0,
    avg_reset=100,
)
