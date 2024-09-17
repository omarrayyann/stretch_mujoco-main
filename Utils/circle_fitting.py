from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np 

def fit_circle_2d(x, y, w=[]):
    
    A = array([x, y, ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = diag(w)
        A = dot(W,A)
        b = dot(W,b)
    
    # Solve by method of least squares
    c = linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def rodrigues_rot(P, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/linalg.norm(n0)
    n1 = n1/linalg.norm(n1)
    k = cross(n0,n1)
    k = k/linalg.norm(k)
    theta = arccos(dot(n0,n1))
    
    # Compute rotated points
    P_rot = zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*cos(theta) + cross(k,P[i])*sin(theta) + k*dot(k,P[i])*(1-cos(theta))

    return P_rot

def angle_between(u, v, n=None):
    if n is None:
        return arctan2(linalg.norm(cross(u,v)), dot(u,v))
    else:
        return arctan2(dot(n,cross(u,v)), dot(u,v))

    
def fit_circle_to_points(P):
    
    P_mean = P.mean(axis=0)
    P_centered = P - P_mean
    U,s,V = linalg.svd(P_centered)

    normal = V[2,:]
    d = -dot(P_mean, normal) 
    P_xy = rodrigues_rot(P_centered, normal, [0,0,1])

    xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])

    C = rodrigues_rot(array([xc,yc,0]), [0,0,1], normal) + P_mean
    C = C.flatten()

    # Check direction 
    if not is_clockwise(P,C,normal):
        normal = -normal

    return C, r, normal


def is_clockwise(points, center, normal):

    clock = 0
    counter = 0

    for i in range(len(points)-1):

        point = points[i]
        next_point = points[i+1]

        to_circle = point - center

        vec = np.cross(to_circle, normal)
        vec = vec/np.linalg.norm(vec)
        opp_vec = -vec

        vec_to_next_point = next_point-point
        vec_to_next_point = vec_to_next_point/np.linalg.norm(vec_to_next_point)

        vec_counter_clockwise = np.linalg.norm(vec-vec_to_next_point)
        vec_counter_counterclockwise = np.linalg.norm(opp_vec-vec_to_next_point)

        if vec_counter_clockwise > vec_counter_counterclockwise:
            clock += 1
        else:  
            counter += 1

    return clock >= counter






    # Translate points to the circle's center
    points_centered = points - center
    
    # Find orthonormal basis for the plane
    normal = normal / np.linalg.norm(normal)
    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(normal, [1, 0, 0]) else np.array([0, 1, 0])
    
    U = arbitrary_vector - np.dot(arbitrary_vector, normal) * normal
    U = U / np.linalg.norm(U)
    V = np.cross(normal, U)
    
    # Project points onto the plane
    projected_points = np.array([
        [np.dot(point, U), np.dot(point, V)]
        for point in points_centered
    ])
    
    # Compute vectors between consecutive points
    vectors = np.diff(projected_points, axis=0)
    
    # Calculate cross products for consecutive vectors
    cross_products = np.cross(vectors[:-1], vectors[1:])
    
    # Check if cross products are positive or negative relative to normal
    # Normalize the cross products direction to get sign
    cross_signs = np.sign(cross_products)
    
    # Determine clockwise or counterclockwise based on sum of signs
    if np.sum(cross_signs) > 0:
        return "counterclockwise"
    else:
        return "clockwise"
    


    
    # import numpy as np
# from numpy.linalg import svd, lstsq

# def generate_circle_by_vectors(t, C, r, n, u):
#     n = n / np.linalg.norm(n)
#     u = u / np.linalg.norm(u)
#     P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
#     return P_circle
# import numpy as np
# from numpy.linalg import svd

# def rodrigues_rot(P, axis, target_axis):
#     """
#     Rodrigues' rotation formula to rotate vector P from axis to target_axis.
#     """
#     axis = np.asarray(axis)
#     target_axis = np.asarray(target_axis)
    
#     if np.allclose(axis, target_axis):
#         return P
#     if np.allclose(axis, -target_axis):
#         return -P
    
#     axis = axis / np.linalg.norm(axis)
#     target_axis = target_axis / np.linalg.norm(target_axis)
#     v = np.cross(axis, target_axis)
#     c = np.dot(axis, target_axis)
#     s = np.linalg.norm(v)
#     k_mat = np.array([[0, -v[2], v[1]],
#                       [v[2], 0, -v[0]],
#                       [-v[1], v[0], 0]])
    
#     R = np.eye(3) + k_mat + k_mat @ k_mat * ((1 - c) / (s ** 2))
    
#     P_rotated = np.dot(P, R.T)
#     return P_rotated

# def fit_circle_2d(x, y):
#     """
#     Fit a circle to 2D points using algebraic distance minimization.
#     """
#     A = np.c_[x, y, np.ones(x.shape)]
#     b = x**2 + y**2
#     c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#     xc, yc = c[0] / 2, c[1] / 2
#     r = np.sqrt(c[2] + xc**2 + yc**2)
#     return xc, yc, r

# def fit_circle_to_points(points):
#     if len(points) < 3:
#         return None, None, None

#     P_mean = points.mean(axis=0)
#     P_centered = points - P_mean
#     U, s, Vt = svd(P_centered)
#     normal = Vt[2, :]

#     # Rotate points into the plane defined by the normal vector
#     P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

#     # Calculate cumulative cross product to determine overall orientation
#     cumulative_cross_product_z = 0
#     for i in range(len(P_xy) - 1):
#         p1 = P_xy[i]
#         p2 = P_xy[i + 1]
#         cross_product_z = np.cross(p2 - p1, -p1)[2]
#         cumulative_cross_product_z += cross_product_z

#     # Check orientation and adjust normal if needed
#     if cumulative_cross_product_z < 0:  # Points are mostly counterclockwise
#         normal = -normal  # Flip normal
#         P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

#     xc, yc, r_fitted = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

#     # Transform the fitted circle center back to 3D space
#     C_fitted = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
#     C_fitted = C_fitted.flatten()

#     return C_fitted, r_fitted, normal

# def set_axes_equal_3d(ax):
#     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
#     spans = abs(limits[:, 0] - limits[:, 1])
#     centers = np.mean(limits, axis=1)
#     radius = 0.5 * max(spans)
#     ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
#     ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
#     ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

# def perform_circle_fitting_limited_points(num_points, C_actual, r, theta, phi, noise_level):
#     t_vals = np.linspace(-np.pi, -0.25 * np.pi, num_points)
#     P_test = generate_circle_by_vectors(
#         t_vals, C_actual, r, np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]),
#         np.array([-np.sin(phi), np.cos(phi), 0])
#     )
#     P_test += np.random.normal(size=P_test.shape) * noise_level
#     P_mean = P_test.mean(axis=0)
#     P_centered = P_test - P_mean
#     U, s, Vt = svd(P_centered)
#     normal = Vt[2, :]
#     P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
#     xc, yc, r_fitted = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
#     C_fitted = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
#     C_fitted = C_fitted.flatten()
#     return P_test, C_fitted, r_fitted, normal

# import numpy as np
# from numpy.linalg import svd

# def fit_circle_to_points(points):
#     if len(points) < 3:
#         return None, None, None

#     P_mean = points.mean(axis=0)
#     P_centered = points - P_mean
#     U, s, Vt = svd(P_centered)
#     normal = Vt[2, :]

#     # Rotate points into the plane defined by the normal vector
#     P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

#     xc, yc, r_fitted = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

#     # Transform the fitted circle center back to 3D space
#     C_fitted = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
#     C_fitted = C_fitted.flatten()

#     return C_fitted, r_fitted, normal

# def is_clockwise(points_2d):
#     """
#     Determines if the set of 2D points are in clockwise order.
#     Expects points_2d to be an Nx2 array.
#     """
#     sum = 0.0
#     for i in range(len(points_2d)-1):
#         x1, y1 = points_2d[i]
#         x2, y2 = points_2d[(i + 1) ]
#         sum += (x2 - x1) * (y2 + y1)
#     return sum > 0



# # def fit_circle_to_points(points):

# #     print(points)
# #     if len(points) < 3:
# #         return None, None, None

# #     P_mean = points.mean(axis=0)
# #     P_centered = points - P_mean
# #     U, s, Vt = svd(P_centered)
# #     normal = Vt[2, :]
# #     d = -np.dot(P_mean, normal)

# #     P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

# #     xc, yc, r_fitted = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])

# #     C_fitted = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
# #     C_fitted = C_fitted.flatten()

# #     # Check if points are in clockwise order
# #     v1 = P_xy[1] - P_xy[0]
# #     v2 = P_xy[2] - P_xy[1]
# #     cross_product = np.cross(v1, v2)

# #     if cross_product[2] < 0:  # Check the z-component of the cross product
# #         # Points are in counterclockwise order, so flip the normal
# #         normal = -normal

# #     return C_fitted, r_fitted, normal

