import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合保存图片
import matplotlib.pyplot as plt

# 设置支持中文和负号的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def buffon_needle(num_trials=10000, needle_length=1.0, line_spacing=1.0):
    hits = 0  # 相交次数

    for _ in range(num_trials):
        y = np.random.uniform(0, line_spacing / 2)
        theta = np.random.uniform(0, np.pi / 2)
        if (needle_length / 2) * np.sin(theta) >= y:
            hits += 1

    if hits == 0:
        return None  # 防止除以 0

    pi_estimate = (2 * needle_length * num_trials) / (line_spacing * hits)
    return pi_estimate

# 增加更多试验点
trial_points = [100, 300, 500, 1000, 3000, 5000, 10000, 30000, 50000, 100000]
for trials in trial_points:
    estimate = buffon_needle(num_trials=trials)
    print(f"试验次数：{trials:6d}，估计的 π ≈ {estimate:.6f}")

true_pi = np.pi
errors = []

for n in trial_points:
    pi_est = buffon_needle(num_trials=n)
    error = abs(pi_est - true_pi)
    errors.append(error)

plt.figure(figsize=(10, 6))
plt.plot(trial_points, errors, marker='o', linestyle='-', color='#0072B2', linewidth=2, markersize=8, label='估计误差')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('试验次数（log）', fontsize=14)
plt.ylabel('估计误差 |π估计 - π真实|（log）', fontsize=14)
plt.title('Buffon 投针实验：实验次数 vs π估计误差', fontsize=16, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\31025\OneDrive\桌面\t\buffon_pi_error.png', dpi=300)
# plt.show()  # 非交互式后端下无需显示