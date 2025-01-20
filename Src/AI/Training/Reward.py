
class Reward:
    def __init__(self, cookies, agent):
        self.cookies = cookies
        self.agent = agent

    def apply_reward(self):
        self.agent.reward(self.cookies)
