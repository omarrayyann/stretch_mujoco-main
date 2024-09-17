import Utils
import numpy as np

class LinearSampler:

    def __init__(self):
        
        self.centers = []
        self.direction = []
        self.rewards = []
        self.total_rewards = [0,0,0,0,0,0]

    def update(self, reward, vector):

        vector = vector/np.linalg.norm(vector)

        if vector[0]>0:
            self.total_rewards[0] += reward*abs(vector[0])
        else:
            self.total_rewards[1] += reward*abs(vector[0])

        if vector[1]>0:
            self.total_rewards[2] += reward*abs(vector[1])
        else:
            self.total_rewards[3] += reward*abs(vector[1])

        if vector[2]>0:
            self.total_rewards[4] += reward*abs(vector[2])
        else:
            self.total_rewards[5] += reward*abs(vector[2])
    
    def sample(self):

        vector = np.array([ self.total_rewards[0]-self.total_rewards[1] , self.total_rewards[2]-self.total_rewards[3] , self.total_rewards[4]-self.total_rewards[5] ])
        vector_magnitude = np.linalg.norm(vector)
        vector = vector/vector_magnitude

        return vector

    
