path = 'scripts/run_evaluation.py'
with open(path, 'r') as f:
    c = f.read()
c = c.replace('KalmanVelocityEstimator', 'KalmanFilterEstimator')
with open(path, 'w') as f:
    f.write(c)
print('fixed')
