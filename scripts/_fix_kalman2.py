path = 'scripts/run_evaluation.py'
with open(path, 'r') as f:
    c = f.read()

old = '''def velocity_kalman(positions, timestamps):
    """Kalman 滤波速度估计"""
    from src.trajectory.velocity_estimator import KalmanFilterEstimator
    dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.002
    est = KalmanFilterEstimator(dt=dt, pos_noise_std=0.005)
    vels = []
    for pos in positions:
        result = est.update(pos)
        vels.append(result["velocity"].copy())
    return np.array(vels)'''

new = '''def velocity_kalman(positions, timestamps):
    """Kalman 滤波速度估计（批量）"""
    from src.trajectory.velocity_estimator import KalmanFilterEstimator
    est = KalmanFilterEstimator(dim=positions.shape[1] if positions.ndim > 1 else 1)
    velocities, _ = est.estimate(positions, timestamps)
    return velocities'''

assert old in c, 'old kalman fn not found'
c = c.replace(old, new)

with open(path, 'w') as f:
    f.write(c)
print('fixed kalman')
