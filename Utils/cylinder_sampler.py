import Utils
import numpy as np

class Cylinder_Sampler:

    def __init__(self):
        self.centers = []
        self.normals = []
        self.rewards = []

    def update(self, center, normal, reward):
        print("reward: ", reward)
        self.centers.append(center)
        self.normals.append(normal)
        self.rewards.append(reward)
            
    def sample(self):

        optimal_center = np.array([0.0, 0, 0.0])
        optimal_normal = np.array([0.0, 0.0, 0.0])
        total_reward = 0.0

        for i in range(len(self.centers)):
            optimal_center += np.array(self.centers[i])*self.rewards[i]
            optimal_normal += np.array(self.normals[i])*self.rewards[i]
            total_reward += self.rewards[i]
        
        optimal_center /= total_reward
        optimal_normal /= total_reward

        return optimal_center, optimal_normal


def generate_cylinder_trajectory(start_point, normal_vector, circle_center, delta_length, total_length):

    closest_point = closest_point_on_line(circle_center,normal_vector,start_point)

    return Utils.generate_circle_trajectory(start_point, normal_vector, closest_point, delta_length, total_length), closest_point


def closest_point_on_line(point_on_line, normal_vector, point):
    x0, y0, z0 = point_on_line
    a, b, c = normal_vector
    x, y, z = point
    t = (a * (x - x0) + b * (y - y0) + c * (z - z0)) / (a**2 + b**2 + c**2)    
    closest_point = (x0 + t * a, y0 + t * b, z0 + t * c)
    return closest_point