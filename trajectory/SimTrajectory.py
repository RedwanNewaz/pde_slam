import numpy as np

LANDMARKS = np.array(
    [
        [30.0, 15.0],
        [30.0, 25.0],
        [33.0, 37.0],
        [38.0, 32.0],
        [34.0, 22.0],
        [26.0, 12.0],
        [26.0, 28.0],
        [32.0, 9.0],
    ]
)

# LANDMARKS = np.array([
#     [20.0, 10.0],
#     [10.0, 7.5],
#     [15.0, 0.5],
#     [25.0, 5.0],
#     [30.0, 15.0],
# ])
#
# LANDMARKS = np.array([
#     [20.0, 12.2],
#     [6.56, 7.25],
#     [15.0, 2.64],
#     [25.0, 2.27],
#     [30.0, 10.0],
# ])


def get_ellipse_points():
    """
    Generate points on an ellipse centered at the origin.

    Parameters:
    a (float): Semi-major axis length.
    b (float): Semi-minor axis length.
    num_points (int): Number of points to generate along the ellipse.

    Returns:
    np.ndarray: Array of shape (num_points, 2) containing (x, y) coordinates of points on the ellipse.
    """
    num_points = 500  # number of waypoints on ellipse
    cx, cy = 25.0, 20.0  # center
    a = 15.0 / 2.0  # semi-major axis
    b = 10.0 / 3.0  # semi-minor axis

    # parameter values (equivalent to i = 0..numPoints)
    theta = np.linspace(0.0, 2.0 * np.pi, num_points + 1)

    x = cx + a * np.cos(theta)
    y = cy + b * np.sin(theta)

    # shape: (num_points+1, 2)
    waypoints = np.column_stack((x, y))
    return waypoints


def gen_control_inputs(waypoints, dt):
    """
    Generate control inputs (velocities) to follow the given waypoints.

    Parameters:
    waypoints (np.ndarray): Array of shape (N, 2) containing (x, y) coordinates of waypoints.
    dt (float): Time step between waypoints.

    Returns:
    np.ndarray: Array of shape (N-1, 2) containing (vx, vy) velocities to move between waypoints.
    """
    dx = np.diff(waypoints[:, 0])
    dy = np.diff(waypoints[:, 1])

    theta = np.arctan2(dy, dx)
    theta = np.unwrap(theta)

    ds = np.sqrt(dx**2 + dy**2)
    v = ds / dt

    omega = np.diff(theta) / dt

    v = v[:-1]
    u = np.column_stack((v, omega))
    return u


def motion_model(x, u, dt):
    F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])

    B = np.array(
        [[dt * np.cos(x[2, 0]), 0], [dt * np.sin(x[2, 0]), 0], [0.0, dt], [1.0, 0.0]]
    )

    x = F.dot(x) + B.dot(u)

    return x


if __name__ == "__main__":
    waypoints = get_ellipse_points()
    dt = 0.1
    u = gen_control_inputs(waypoints, dt)
    # State Vector [x y yaw v]'
    x_est = np.zeros((4, 1))
    x_est[0, 0] = waypoints[0, 0]
    x_est[1, 0] = waypoints[0, 1]
    x_est[2, 0] = np.pi / 2.0
    trajectory = []
    for i in range(u.shape[0]):
        x_est = motion_model(x_est, u[i, :].reshape(2, 1), dt)
        trajectory.append(x_est.flatten())
    trajectory = np.array(trajectory)
    # np.savetxt("sim_trajectory.txt", trajectory)

    # For visualization
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(waypoints[:, 0], waypoints[:, 1], "r--", label="Waypoints")
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", label="Trajectory")
    plt.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "k*", markersize=10, label="Landmarks")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.title("Simulated Trajectory Following Elliptical Waypoints")
    plt.grid()
    plt.show()
