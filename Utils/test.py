import numpy as np
from numpy.linalg import svd, lstsq

def generate_circle_by_vectors(t, C, r, n, u):
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
    return P_circle

def rodrigues_rot(P, n0, n1):
    if P.ndim == 1:
        P = P[np.newaxis, :]
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))
    return P_rot

def fit_circle_2d(x, y, w=[]):
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)
    c = lstsq(A, b, rcond=None)[0]
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r

def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = abs(limits[:, 0] - limits[:, 1])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

def perform_circle_fitting_limited_points(num_points, C_actual, r, theta, phi, noise_level):
    t_vals = np.linspace(-np.pi, -0.25 * np.pi, num_points)
    P_test = generate_circle_by_vectors(
        t_vals, C_actual, r, np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]),
        np.array([-np.sin(phi), np.cos(phi), 0])
    )
    P_test += np.random.normal(size=P_test.shape) * noise_level
    P_mean = P_test.mean(axis=0)
    P_centered = P_test - P_mean
    U, s, Vt = svd(P_centered)
    normal = Vt[2, :]
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
    xc, yc, r_fitted = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
    C_fitted = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C_fitted = C_fitted.flatten()
    return P_test, C_fitted, r_fitted, normal
