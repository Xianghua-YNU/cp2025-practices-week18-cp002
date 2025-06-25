import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义常数
a = 5.29e-2  # 波尔半径，单位：nm
D_max = 1.1  # 概率密度最大值，用于拒绝采样上限
r0 = 0.25    # 采样半径上限（收敛半径），单位：nm
n_points = 20000  # 要采样的电子数量

# 概率密度函数 D(r)
def D_r(r):
    return (4 * r**2 / a**3) * np.exp(-2 * r / a)

# 拒绝采样获取半径 r
def sample_radius(n_points):
    r_list = []
    while len(r_list) < n_points:
        r_candidate = np.random.uniform(0, r0)
        y = np.random.uniform(0, D_max)
        if y < D_r(r_candidate):
            r_list.append(r_candidate)
    return np.array(r_list)

# 生成三维电子坐标
def sample_electron_positions(n_points):
    r = sample_radius(n_points)
    theta = np.arccos(1 - 2 * np.random.rand(n_points))  # θ ∈ [0, π]
    phi = 2 * np.pi * np.random.rand(n_points)           # φ ∈ [0, 2π]

    # 球坐标转直角坐标
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# 可视化电子云
def visualize_electron_cloud(x, y, z):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.3, color='blue')
    ax.set_title("氢原子基态电子云分布（n=1, l=0, m=0）", fontsize=14)
    ax.set_xlabel("x 方向（纳米）")
    ax.set_ylabel("y 方向（纳米）")
    ax.set_zlabel("z 方向（纳米）")
    plt.tight_layout()
    plt.show()

# 主程序入口
if __name__ == "__main__":
    x, y, z = sample_electron_positions(n_points)
    visualize_electron_cloud(x, y, z)
