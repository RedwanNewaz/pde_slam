from SimTrajectory import *
import matplotlib.pyplot as plt

# Estimation parameter of PF
Q = np.diag([0.2]) ** 2  # range error
R = np.diag([2.0, np.deg2rad(40.0)]) ** 2  # input error

#  Simulation parameter
Q_sim = np.diag([0.2]) ** 2
R_sim = np.diag([1.0, np.deg2rad(30.0)]) ** 2

dt = 0.1  # time tick [s]

# Particle filter parameter
NP = 100  # Number of Particle


def particle_motion(px, u):
    for ip in range(NP):
        x = np.array([px[:, ip]]).T

        #  Predict with random input sampling
        ud1 = u[0, 0] + np.random.randn() * R[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * R[1, 1] ** 0.5
        ud = np.array([[ud1, ud2]]).T
        x = motion_model(x, ud, dt)
        px[:, ip] = x[:, 0]
    return px

if __name__ == "__main__":
    waypoints = get_ellipse_points()
    u = gen_control_inputs(waypoints, dt)
    # State Vector [x y yaw v]'
    x_est = np.zeros((4, 1))
    x_est[0, 0] = waypoints[0, 0]
    x_est[1, 0] = waypoints[0, 1]
    x_est[2, 0] = np.pi / 2.0
    trajectory = np.zeros((0, 4))

    px = np.zeros((4, NP))  # Particle store
    for ip in range(NP):
        px[0, ip] = x_est[0, 0] + np.random.randn() * R[0, 0] ** 0.5
        px[1, ip] = x_est[1, 0] + np.random.randn() * R[1, 1] ** 0.5
        px[2, ip] = x_est[2, 0] + np.random.randn() * np.deg2rad(10.0)
        px[3, ip] = x_est[3, 0] + np.random.randn() * 1.0



    for i in range(u.shape[0]):
        x_est = motion_model(x_est, u[i, :].reshape(2, 1), dt)
        plt.clf()
        plt.plot(waypoints[:, 0], waypoints[:, 1], 'r--', label='Waypoints')
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Trajectory')

        px = particle_motion(px, u[i, :].reshape(2, 1))

        plt.plot(px[0, :], px[1, :], ".g")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')

        plt.pause(0.001)

        trajectory = np.vstack((trajectory, np.squeeze(x_est)))
    trajectory = np.array(trajectory)