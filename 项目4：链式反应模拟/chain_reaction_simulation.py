import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class ChainReactionSimulator:
    def __init__(self, branching_factor=1.5, termination_prob=0.1, max_generations=20):
        """
        初始化链式反应模拟器
        
        参数:
        - branching_factor: 平均每个粒子产生的次级粒子数
        - termination_prob: 反应终止概率
        - max_generations: 最大模拟代数
        """
        self.branching_factor = branching_factor
        self.termination_prob = termination_prob
        self.max_generations = max_generations
        
    def simulate(self, num_simulations=1000):
        """
        运行多次模拟并收集统计数据
        
        参数:
        - num_simulations: 模拟次数
        
        返回:
        - 各代粒子数的平均值和标准差
        - 所有模拟的终止代数
        """
        generation_stats = defaultdict(list)
        termination_generations = []
        
        for _ in range(num_simulations):
            generations, particles_per_gen = self._single_simulation()
            termination_generations.append(generations)
            
            for gen, count in enumerate(particles_per_gen):
                generation_stats[gen].append(count)
                
        # 计算各代的平均值和标准差
        avg_particles = []
        std_particles = []
        max_gen = max(generation_stats.keys()) if generation_stats else 0
        
        for gen in range(max_gen + 1):
            counts = generation_stats[gen]
            avg_particles.append(np.mean(counts))
            std_particles.append(np.std(counts))
            
        return avg_particles, std_particles, termination_generations
    
    def _single_simulation(self):
        """
        单次模拟运行
        
        返回:
        - 总代数
        - 各代粒子数列表
        """
        current_particles = 1  # 初始粒子
        particles_per_gen = [current_particles]
        generations = 0
        
        while current_particles > 0 and generations < self.max_generations:
            next_gen_particles = 0
            
            for _ in range(current_particles):
                # 检查是否终止
                if np.random.random() < self.termination_prob:
                    continue
                
                # 产生次级粒子数 (泊松分布)
                new_particles = np.random.poisson(self.branching_factor)
                next_gen_particles += new_particles
                
            particles_per_gen.append(next_gen_particles)
            current_particles = next_gen_particles
            generations += 1
            
        return generations, particles_per_gen
    
    def plot_results(self, avg_particles, std_particles):
        """绘制模拟结果"""
        generations = range(len(avg_particles))
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(generations, avg_particles, yerr=std_particles, 
                     fmt='-o', capsize=5, label='平均粒子数±标准差')
        plt.xlabel('代数')
        plt.ylabel('粒子数')
        plt.title(f'链式反应模拟 (分支因子={self.branching_factor}, 终止概率={self.termination_prob})')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def analyze_termination(self, termination_generations):
        """分析终止代数分布"""
        plt.figure(figsize=(10, 6))
        plt.hist(termination_generations, bins=range(max(termination_generations)+2), 
                 alpha=0.7, edgecolor='black')
        plt.xlabel('终止代数')
        plt.ylabel('频数')
        plt.title(f'链式反应终止代数分布 (分支因子={self.branching_factor}, 终止概率={self.termination_prob})')
        plt.grid(True)
        plt.show()
        
        print(f"平均终止代数: {np.mean(termination_generations):.2f}")
        print(f"终止代数标准差: {np.std(termination_generations):.2f}")

# 参数敏感性分析
def parameter_sensitivity_analysis():
    """分析不同参数对链式反应结果的影响"""
    # 测试不同分支因子
    branching_factors = [0.8, 1.0, 1.2, 1.5, 2.0]
    termination_prob = 0.1
    
    plt.figure(figsize=(10, 6))
    for bf in branching_factors:
        simulator = ChainReactionSimulator(branching_factor=bf, termination_prob=termination_prob)
        avg_particles, _, _ = simulator.simulate(num_simulations=500)
        plt.plot(avg_particles, '-o', label=f'分支因子={bf}')
    
    plt.xlabel('代数')
    plt.ylabel('平均粒子数')
    plt.title(f'不同分支因子对链式反应的影响 (终止概率={termination_prob})')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 测试不同终止概率
    branching_factor = 1.5
    termination_probs = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    plt.figure(figsize=(10, 6))
    for tp in termination_probs:
        simulator = ChainReactionSimulator(branching_factor=branching_factor, termination_prob=tp)
        avg_particles, _, _ = simulator.simulate(num_simulations=500)
        plt.plot(avg_particles, '-o', label=f'终止概率={tp}')
    
    plt.xlabel('代数')
    plt.ylabel('平均粒子数')
    plt.title(f'不同终止概率对链式反应的影响 (分支因子={branching_factor})')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 基本模拟
    simulator = ChainReactionSimulator(branching_factor=1.5, termination_prob=0.1)
    avg_particles, std_particles, term_gens = simulator.simulate(num_simulations=1000)
    simulator.plot_results(avg_particles, std_particles)
    simulator.analyze_termination(term_gens)
    
    # 参数敏感性分析
    parameter_sensitivity_analysis()
