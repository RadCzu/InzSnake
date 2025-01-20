import numpy as np


class LearningExperience:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
        self.position = 0
        self.previous_state = None

    def add_experience(self, experience):
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.max_size

    def add_deep_experience(self, action, reward, state, done):
        if self.previous_state is not None:
            experience = (self.previous_state, action, reward, state, done)
            self.add_experience(experience)
        self.previous_state = state

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        return zip(*batch)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def count_moves(self, data=None):
        """
        Counts the moves for each direction from the memory or the provided data.

        Parameters:
            data: Optional. A list of experiences to analyze. Defaults to the full memory.

        Returns:
            A list with the count of moves in each direction [left, right, up, down].
        """
        if data is None:
            data = self.memory

        direction_counts = [0, 0, 0, 0]  # Indices for [left, right, up, down]
        for experience in data:
            _, action, _, _, _ = experience  # Unpack experience
            move_direction = action.index(max(action))  # Find the index of the direction
            direction_counts[move_direction] += 1

        print(f"Move counts: Left={direction_counts[0]}, Right={direction_counts[1]}, "
              f"Up={direction_counts[2]}, Down={direction_counts[3]}")
        return direction_counts
