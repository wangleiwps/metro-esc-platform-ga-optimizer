"""
遗传算法 (Genetic Algorithm) 模块

用于地铁疏散平台钢支架长度优化的遗传算法实现。
支持动态安全余量计算和成本敏感聚类。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import random
from dataclasses import dataclass


@dataclass
class GAConfig:
    """遗传算法配置参数"""
    population_size: int = 100
    max_generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_rate: float = 0.1
    tournament_size: int = 5
    convergence_threshold: float = 1e-6
    max_stagnation: int = 50


@dataclass
class SteelSupport:
    """钢支架数据结构"""
    id: str
    required_length: float
    measurement_error: float
    geometric_complexity: float
    construction_condition: float


class DynamicSafetyMargin:
    """动态安全余量计算器"""
    
    def __init__(self, base_margin: float = 30.0):
        """
        初始化动态安全余量计算器
        
        Args:
            base_margin: 基础安全余量 (mm)
        """
        self.base_margin = base_margin
    
    def calculate_margin(self, 
                        measurement_precision: float,
                        geometric_complexity: float, 
                        construction_condition: float,
                        safety_factor: float = 1.2) -> float:
        """
        计算动态安全余量
        
        公式: Δ = σ × δ × ε × f
        其中:
        - σ: 测量精度因子
        - δ: 几何复杂度因子  
        - ε: 施工条件因子
        - f: 安全系数
        
        Args:
            measurement_precision: 测量精度因子 (0.5-2.0)
            geometric_complexity: 几何复杂度因子 (0.8-1.5)
            construction_condition: 施工条件因子 (0.9-1.3)
            safety_factor: 安全系数 (1.0-1.5)
            
        Returns:
            动态安全余量 (mm)
        """
        dynamic_margin = (self.base_margin * 
                         measurement_precision * 
                         geometric_complexity * 
                         construction_condition * 
                         safety_factor)
        
        return max(10.0, min(100.0, dynamic_margin))  # 限制在10-100mm范围内


class CostSensitiveClustering:
    """成本敏感聚类算法"""
    
    def __init__(self, material_cost_per_mm: float = 0.15):
        """
        初始化成本敏感聚类器
        
        Args:
            material_cost_per_mm: 每毫米材料成本 (元/mm)
        """
        self.material_cost_per_mm = material_cost_per_mm
        self.standardization_cost = 50.0  # 标准化成本 (元/种规格)
    
    def calculate_total_cost(self, lengths: List[float], quantities: List[int]) -> float:
        """
        计算总成本 = 材料成本 + 标准化成本 + 浪费成本
        
        Args:
            lengths: 标准长度列表
            quantities: 对应数量列表
            
        Returns:
            总成本 (元)
        """
        # 材料成本
        material_cost = sum(l * q * self.material_cost_per_mm for l, q in zip(lengths, quantities))
        
        # 标准化成本
        standardization_cost = len(lengths) * self.standardization_cost
        
        # 浪费成本 (简化计算)
        waste_cost = sum(max(0, l - min(lengths)) * q * 0.1 for l, q in zip(lengths, quantities))
        
        return material_cost + standardization_cost + waste_cost
    
    def improved_kmeans(self, data: List[float], k: int, max_iters: int = 100) -> Tuple[List[float], List[int]]:
        """
        改进的K-means聚类算法，考虑成本因素
        
        Args:
            data: 需要聚类的长度数据
            k: 聚类数量
            max_iters: 最大迭代次数
            
        Returns:
            (聚类中心, 聚类标签)
        """
        if len(data) <= k:
            return data, list(range(len(data)))
        
        # 初始化聚类中心
        centers = random.sample(data, k)
        
        for _ in range(max_iters):
            # 分配数据点到最近的聚类中心
            labels = []
            for point in data:
                distances = [abs(point - center) for center in centers]
                labels.append(distances.index(min(distances)))
            
            # 更新聚类中心
            new_centers = []
            for i in range(k):
                cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
                if cluster_points:
                    new_centers.append(sum(cluster_points) / len(cluster_points))
                else:
                    new_centers.append(centers[i])
            
            # 检查收敛
            if all(abs(new - old) < 1e-3 for new, old in zip(new_centers, centers)):
                break
                
            centers = new_centers
        
        return centers, labels


class GeneticAlgorithm:
    """遗传算法主类"""
    
    def __init__(self, config: GAConfig):
        """
        初始化遗传算法
        
        Args:
            config: GA配置参数
        """
        self.config = config
        self.safety_margin = DynamicSafetyMargin()
        self.clustering = CostSensitiveClustering()
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def create_individual(self, steel_supports: List[SteelSupport]) -> List[float]:
        """
        创建个体 (染色体)
        
        Args:
            steel_supports: 钢支架列表
            
        Returns:
            个体 (标准长度列表)
        """
        # 计算每个钢支架的最小所需长度 (包含动态安全余量)
        min_lengths = []
        for support in steel_supports:
            margin = self.safety_margin.calculate_margin(
                measurement_precision=1.0 + support.measurement_error,
                geometric_complexity=support.geometric_complexity,
                construction_condition=support.construction_condition
            )
            min_length = support.required_length + margin
            min_lengths.append(min_length)
        
        # 使用聚类算法生成初始标准长度
        k = min(10, len(set(min_lengths)))  # 最多10种规格
        if k > 1:
            centers, _ = self.clustering.improved_kmeans(min_lengths, k)
            return sorted(centers)
        else:
            return min_lengths
    
    def evaluate_fitness(self, individual: List[float], steel_supports: List[SteelSupport]) -> float:
        """
        评估个体适应度 (成本越低适应度越高)
        
        Args:
            individual: 个体 (标准长度列表)
            steel_supports: 钢支架列表
            
        Returns:
            适应度值
        """
        # 计算每个钢支架应该使用的标准长度
        assignments = []
        for support in steel_supports:
            margin = self.safety_margin.calculate_margin(
                measurement_precision=1.0 + support.measurement_error,
                geometric_complexity=support.geometric_complexity,
                construction_condition=support.construction_condition
            )
            min_required = support.required_length + margin
            
            # 选择满足要求的最小标准长度
            suitable_lengths = [length for length in individual if length >= min_required]
            if suitable_lengths:
                assignments.append(min(suitable_lengths))
            else:
                # 如果没有合适的长度，使用最大的标准长度并添加惩罚
                assignments.append(max(individual))
        
        # 计算每种标准长度的使用数量
        length_counts = {}
        for length in assignments:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 计算总成本
        lengths = list(length_counts.keys())
        quantities = list(length_counts.values())
        total_cost = self.clustering.calculate_total_cost(lengths, quantities)
        
        # 适应度 = 1 / (1 + 总成本)
        return 1.0 / (1.0 + total_cost)
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """
        交叉操作
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            (子代1, 子代2)
        """
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 单点交叉
        min_len = min(len(parent1), len(parent2))
        if min_len <= 1:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, min_len - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return sorted(child1), sorted(child2)
    
    def mutate(self, individual: List[float]) -> List[float]:
        """
        变异操作
        
        Args:
            individual: 个体
            
        Returns:
            变异后的个体
        """
        if random.random() > self.config.mutation_rate:
            return individual.copy()
        
        mutated = individual.copy()
        
        if len(mutated) > 0:
            # 随机选择一个基因进行变异
            idx = random.randint(0, len(mutated) - 1)
            # 在原值基础上添加小的随机扰动
            perturbation = random.gauss(0, 10)  # 标准差为10mm的高斯扰动
            mutated[idx] = max(100, mutated[idx] + perturbation)  # 最小长度100mm
        
        return sorted(mutated)
    
    def tournament_selection(self, population: List[List[float]], 
                           fitness_scores: List[float]) -> List[float]:
        """
        锦标赛选择
        
        Args:
            population: 种群
            fitness_scores: 适应度分数
            
        Returns:
            选中的个体
        """
        tournament_indices = random.sample(range(len(population)), 
                                         min(self.config.tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx].copy()
    
    def optimize(self, steel_supports: List[SteelSupport]) -> Dict:
        """
        执行遗传算法优化
        
        Args:
            steel_supports: 钢支架列表
            
        Returns:
            优化结果字典
        """
        # 初始化种群
        population = [self.create_individual(steel_supports) 
                     for _ in range(self.config.population_size)]
        
        best_individual = None
        best_fitness = -1
        stagnation_count = 0
        
        for generation in range(self.config.max_generations):
            # 评估适应度
            fitness_scores = [self.evaluate_fitness(ind, steel_supports) 
                            for ind in population]
            
            # 记录最佳和平均适应度
            current_best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # 更新最佳个体
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[fitness_scores.index(current_best_fitness)].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # 检查收敛条件
            if stagnation_count >= self.config.max_stagnation:
                break
            
            # 选择、交叉、变异生成新种群
            new_population = []
            
            # 精英保留
            elite_count = int(self.config.population_size * self.config.elite_rate)
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 生成剩余个体
            while len(new_population) < self.config.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 截断到指定大小
            population = new_population[:self.config.population_size]
        
        # 生成最终结果
        return self._generate_result(best_individual, steel_supports, best_fitness)
    
    def _generate_result(self, best_individual: List[float], 
                        steel_supports: List[SteelSupport], 
                        best_fitness: float) -> Dict:
        """
        生成优化结果
        
        Args:
            best_individual: 最佳个体
            steel_supports: 钢支架列表
            best_fitness: 最佳适应度
            
        Returns:
            结果字典
        """
        # 计算每个钢支架的分配
        assignments = []
        for support in steel_supports:
            margin = self.safety_margin.calculate_margin(
                measurement_precision=1.0 + support.measurement_error,
                geometric_complexity=support.geometric_complexity,
                construction_condition=support.construction_condition
            )
            min_required = support.required_length + margin
            
            suitable_lengths = [length for length in best_individual if length >= min_required]
            if suitable_lengths:
                assigned_length = min(suitable_lengths)
            else:
                assigned_length = max(best_individual)
            
            assignments.append({
                'support_id': support.id,
                'required_length': support.required_length,
                'safety_margin': margin,
                'assigned_length': assigned_length,
                'waste': assigned_length - support.required_length
            })
        
        # 统计标准长度使用情况
        length_counts = {}
        for assignment in assignments:
            length = assignment['assigned_length']
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 计算成本
        lengths = list(length_counts.keys())
        quantities = list(length_counts.values())
        total_cost = self.clustering.calculate_total_cost(lengths, quantities)
        
        return {
            'standard_lengths': sorted(best_individual),
            'assignments': assignments,
            'length_quantities': dict(zip(lengths, quantities)),
            'total_cost': total_cost,
            'fitness': best_fitness,
            'num_standards': len(best_individual),
            'total_waste': sum(a['waste'] for a in assignments),
            'fitness_history': {
                'best': self.best_fitness_history,
                'average': self.avg_fitness_history
            }
        }


def create_sample_data(num_supports: int = 50) -> List[SteelSupport]:
    """
    创建示例钢支架数据
    
    Args:
        num_supports: 钢支架数量
        
    Returns:
        钢支架列表
    """
    supports = []
    for i in range(num_supports):
        support = SteelSupport(
            id=f"S{i+1:03d}",
            required_length=random.uniform(1000, 3000),  # 1-3米
            measurement_error=random.uniform(0.0, 0.2),   # 0-20%误差
            geometric_complexity=random.uniform(0.8, 1.5), # 复杂度因子
            construction_condition=random.uniform(0.9, 1.3) # 施工条件因子
        )
        supports.append(support)
    
    return supports


if __name__ == "__main__":
    # 示例使用
    print("遗传算法钢支架长度优化示例")
    
    # 创建示例数据
    steel_supports = create_sample_data(30)
    
    # 配置GA参数
    config = GAConfig(
        population_size=50,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    # 运行优化
    ga = GeneticAlgorithm(config)
    result = ga.optimize(steel_supports)
    
    # 输出结果
    print(f"\n优化结果:")
    print(f"标准长度: {result['standard_lengths']}")
    print(f"规格数量: {result['num_standards']}")
    print(f"总成本: {result['total_cost']:.2f} 元")
    print(f"总浪费: {result['total_waste']:.2f} mm")
    print(f"适应度: {result['fitness']:.6f}")

