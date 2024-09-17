import Utils
import numpy as np

class CircleSampler:

    def __init__(self):
        self.centers = []
        self.normals = []
        self.rewards = []

    def update(self, center, normal, reward):
        self.centers.append(center)
        self.normals.append(normal)
        self.rewards.append(reward)
            
    def sample(self, original_position, delta_length=0.025, total_length=0.2):
        
        print(self.centers)
        min_radius = min(np.linalg.norm(np.array(original_position) - np.array(center)) for center in self.centers)

        if delta_length > min_radius:
            delta_length = min_radius / 2.0
        if total_length > 2 * np.pi * min_radius:
            total_length = 2 * np.pi * min_radius / 2.0

        num_steps = int(total_length / delta_length) + 1
        weighted_trajectory = np.zeros((num_steps, 3))
        total_reward = sum(self.rewards)
        
        for center, normal, reward in zip(self.centers, self.normals, self.rewards):
            trajectory = np.array(Utils.generate_circle_trajectory(original_position, normal, center, delta_length, total_length))
            weighted_trajectory += trajectory * reward
        
        averaged_trajectory = weighted_trajectory / total_reward

        c_fitted, r_fitted, normal = Utils.fit_circle_to_points(averaged_trajectory)

        return c_fitted, normal
    

def generate_circle_trajectory(start_point, normal_vector, circle_center, delta_length, total_length):
    x0, y0, z0 = start_point
    cx, cy, cz = circle_center
    
    # Calculate the normal vector and the vector from the circle center to the start point
    n_vec = np.array(normal_vector) / np.linalg.norm(normal_vector)
    r_vec = np.array([x0 - cx, y0 - cy, z0 - cz])
    
    # Project start_point onto the plane of the circle
    distance_from_plane = np.dot(r_vec, n_vec)  # Distance from start_point to the plane
    projection_on_plane = np.array(start_point) - distance_from_plane * n_vec

    # Calculate the radius vector in the plane from the circle center to the projection
    vec_from_center_to_projection = projection_on_plane - np.array(circle_center)
    radius = np.linalg.norm(vec_from_center_to_projection)  # Distance from center to the projection
    
    # Correct the projection point to lie exactly on the circle
    if not np.isclose(radius, 0):
        vec_from_center_to_projection_normalized = vec_from_center_to_projection / radius
        start_point_on_circle = np.array(circle_center) + vec_from_center_to_projection_normalized * radius
    else:
        raise ValueError("Starting point cannot coincide with the circle center.")

    # New radius vector using the corrected start point
    r_vec = start_point_on_circle - np.array(circle_center)
    radius = np.linalg.norm(r_vec)  # Re-calculate radius after correction

    # First orthogonal vector (in the plane of the circle)
    u1 = r_vec / radius

    # Second orthogonal vector (perpendicular to both normal and u1)
    u2 = np.cross(n_vec, u1)
    u2 = u2 / np.linalg.norm(u2)

    # Angle increment per step
    delta_theta = delta_length / radius

    # Trajectory generation
    trajectory = []
    num_steps = int(total_length / delta_length)

    for i in range(num_steps + 1):
        theta = i * delta_theta
        point = np.array(circle_center) + radius * (np.cos(theta) * u1 + np.sin(theta) * u2)
        trajectory.append(point)

    return trajectory