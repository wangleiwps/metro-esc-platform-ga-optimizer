"""
模拟退火算法 (Simulated Annealing) 模块

用于地铁疏散平台钢支架长度优化的模拟退火算法实现。
可与遗传算法结合使用，提供局部搜索能力。
"""

import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .ga import SteelSupport, DynamicSafetyMargin, CostSensitiveClustering


@dataclass
class SAConfig:
    """模拟退火算法配置参数"""
    initial_temperature: float = 1000.0
    final_temperature: float = 0.1
    cooling_rate: float = 0.95
    max_iterations: int = 1000
    max_iterations_per_temp: int = 100
    min_improvement_threshold: float = 1e-6
    max_stagnation: int = 50


class SimulatedAnnealing:
    """模拟退火算法主类"""
    
    def __init__(self, config: SAConfig):
        """
        初始化模拟退火算法
        
        Args:
            config: SA配置参数
        """
        self.config = config
        self.safety_margin = DynamicSafetyMargin()
        self.clustering = CostSensitiveClustering()
        self.temperature_history = []
        self.cost_history = []
        self.acceptance_history = []
    
    def calculate_cost(self, solution: List[float], steel_supports: List[SteelSupport]) -> float:
        """
        计算解的成本 (成本越低越好)
        
        Args:
            solution: 当前解 (标准长度列表)
            steel_supports: 钢支架列表
            
        Returns:
            总成本
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
            
            # 选择满足要求的最小标准长度
            suitable_lengths = [length for length in solution if length >= min_required]
            if suitable_lengths:
                assignments.append(min(suitable_lengths))
            else:
                # 如果没有合适的长度，使用最大的标准长度并添加惩罚
                assignments.append(max(solution))
                # 添加惩罚成本
                penalty = (min_required - max(solution)) * 10  # 惩罚系数
                return float('inf')  # 不可行解
        
        # 计算每种标准长度的使用数量
        length_counts = {}
        for length in assignments:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 计算总成本
        lengths = list(length_counts.keys())
        quantities = list(length_counts.values())
        total_cost = self.clustering.calculate_total_cost(lengths, quantities)
        
        return total_cost
    
    def generate_neighbor(self, current_solution: List[float], 
                         steel_supports: List[SteelSupport]) -> List[float]:
        """
        生成邻域解
        
        Args:
            current_solution: 当前解
            steel_supports: 钢支架列表
            
        Returns:
            邻域解
        """
        neighbor = current_solution.copy()
        
        if len(neighbor) == 0:
            return neighbor
        
        # 随机选择邻域操作
        operation = random.choice(['modify', 'add', 'remove', 'swap'])
        
        if operation == 'modify' and len(neighbor) > 0:
            # 修改一个标准长度
            idx = random.randint(0, len(neighbor) - 1)
            perturbation = random.gauss(0, 20)  # 标准差为20mm的高斯扰动
            neighbor[idx] = max(100, neighbor[idx] + perturbation)
            
        elif operation == 'add' and len(neighbor) < 15:  # 限制最大规格数
            # 添加一个新的标准长度
            # 基于现有长度生成新长度
            if neighbor:
                base_length = random.choice(neighbor)
                new_length = max(100, base_length + random.gauss(0, 50))
                neighbor.append(new_length)
            else:
                # 如果没有现有长度，基于钢支架需求生成
                required_lengths = [s.required_length for s in steel_supports]
                new_length = random.choice(required_lengths) + random.uniform(20, 80)
                neighbor.append(new_length)
                
        elif operation == 'remove' and len(neighbor) > 1:
            # 移除一个标准长度
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
            
        elif operation == 'swap' and len(neighbor) >= 2:
            # 交换两个标准长度的位置 (对排序后的列表意义不大，但保留此操作)
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        
        return sorted(neighbor)
    
    def acceptance_probability(self, current_cost: float, new_cost: float, 
                             temperature: float) -> float:
        """
        计算接受概率
        
        Args:
            current_cost: 当前解的成本
            new_cost: 新解的成本
            temperature: 当前温度
            
        Returns:
            接受概率
        """
        if new_cost < current_cost:
            return 1.0  # 更好的解总是接受
        
        if temperature <= 0:
            return 0.0
        
        # Boltzmann概率
        try:
            prob = math.exp(-(new_cost - current_cost) / temperature)
            return min(1.0, prob)
        except OverflowError:
            return 0.0
    
    def cooling_schedule(self, initial_temp: float, iteration: int) -> float:
        """
        冷却调度
        
        Args:
            initial_temp: 初始温度
            iteration: 当前迭代次数
            
        Returns:
            当前温度
        """
        # 指数冷却
        return initial_temp * (self.config.cooling_rate ** iteration)
    
    def optimize(self, initial_solution: List[float], 
                steel_supports: List[SteelSupport]) -> Dict:
        """
        执行模拟退火优化
        
        Args:
            initial_solution: 初始解
            steel_supports: 钢支架列表
            
        Returns:
            优化结果字典
        """
        # 初始化
        current_solution = initial_solution.copy()
        current_cost = self.calculate_cost(current_solution, steel_supports)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temperature = self.config.initial_temperature
        iteration = 0
        stagnation_count = 0
        accepted_moves = 0
        total_moves = 0
        
        # 清空历史记录
        self.temperature_history = []
        self.cost_history = []
        self.acceptance_history = []
        
        while (temperature > self.config.final_temperature and 
               iteration < self.config.max_iterations and
               stagnation_count < self.config.max_stagnation):
            
            temp_accepted = 0
            temp_total = 0
            
            # 在当前温度下进行多次迭代
            for _ in range(self.config.max_iterations_per_temp):
                # 生成邻域解
                neighbor_solution = self.generate_neighbor(current_solution, steel_supports)
                neighbor_cost = self.calculate_cost(neighbor_solution, steel_supports)
                
                # 计算接受概率
                accept_prob = self.acceptance_probability(current_cost, neighbor_cost, temperature)
                
                temp_total += 1
                total_moves += 1
                
                # 决定是否接受新解
                if random.random() < accept_prob:
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost
                    temp_accepted += 1
                    accepted_moves += 1
                    
                    # 更新最佳解
                    if current_cost < best_cost:
                        best_solution = current_solution.copy()
                        best_cost = current_cost
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1
            
            # 记录历史
            self.temperature_history.append(temperature)
            self.cost_history.append(current_cost)
            acceptance_rate = temp_accepted / temp_total if temp_total > 0 else 0
            self.acceptance_history.append(acceptance_rate)
            
            # 冷却
            iteration += 1
            temperature = self.cooling_schedule(self.config.initial_temperature, iteration)
        
        # 生成最终结果
        return self._generate_result(
            best_solution, steel_supports, best_cost, 
            accepted_moves, total_moves, iteration
        )
    
    def _generate_result(self, best_solution: List[float], 
                        steel_supports: List[SteelSupport],
                        best_cost: float, accepted_moves: int, 
                        total_moves: int, iterations: int) -> Dict:
        """
        生成优化结果
        
        Args:
            best_solution: 最佳解
            steel_supports: 钢支架列表
            best_cost: 最佳成本
            accepted_moves: 接受的移动次数
            total_moves: 总移动次数
            iterations: 迭代次数
            
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
            
            suitable_lengths = [length for length in best_solution if length >= min_required]
            if suitable_lengths:
                assigned_length = min(suitable_lengths)
            else:
                assigned_length = max(best_solution)
            
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
        
        return {
            'standard_lengths': sorted(best_solution),
            'assignments': assignments,
            'length_quantities': dict(zip(length_counts.keys(), length_counts.values())),
            'total_cost': best_cost,
            'num_standards': len(best_solution),
            'total_waste': sum(a['waste'] for a in assignments),
            'iterations': iterations,
            'acceptance_rate': accepted_moves / total_moves if total_moves > 0 else 0,
            'convergence_history': {
                'temperature': self.temperature_history,
                'cost': self.cost_history,
                'acceptance_rate': self.acceptance_history
            }
        }


class HybridGASA:
    """GA-SA混合算法"""
    
    def __init__(self, ga_config, sa_config):
        """
        初始化混合算法
        
        Args:
            ga_config: GA配置
            sa_config: SA配置
        """
        from .ga import GeneticAlgorithm
        
        self.ga = GeneticAlgorithm(ga_config)
        self.sa = SimulatedAnnealing(sa_config)
    
    def optimize(self, steel_supports: List[SteelSupport]) -> Dict:
        """
        执行混合优化
        
        Args:
            steel_supports: 钢支架列表
            
        Returns:
            优化结果字典
        """
        # 第一阶段：使用GA进行全局搜索
        print("第一阶段：遗传算法全局搜索...")
        ga_result = self.ga.optimize(steel_supports)
        
        # 第二阶段：使用SA进行局部优化
        print("第二阶段：模拟退火局部优化...")
        sa_result = self.sa.optimize(ga_result['standard_lengths'], steel_supports)
        
        # 比较结果，选择更好的
        if sa_result['total_cost'] < ga_result['total_cost']:
            final_result = sa_result.copy()
            final_result['optimization_method'] = 'GA+SA'
            final_result['ga_result'] = ga_result
        else:
            final_result = ga_result.copy()
            final_result['optimization_method'] = 'GA'
            final_result['sa_result'] = sa_result
        
        return final_result


def create_initial_solution(steel_supports: List[SteelSupport], 
                          num_standards: int = 8) -> List[float]:
    """
    创建初始解
    
    Args:
        steel_supports: 钢支架列表
        num_standards: 标准长度数量
        
    Returns:
        初始解 (标准长度列表)
    """
    safety_margin = DynamicSafetyMargin()
    
    # 计算所有钢支架的最小所需长度
    min_lengths = []
    for support in steel_supports:
        margin = safety_margin.calculate_margin(
            measurement_precision=1.0 + support.measurement_error,
            geometric_complexity=support.geometric_complexity,
            construction_condition=support.construction_condition
        )
        min_length = support.required_length + margin
        min_lengths.append(min_length)
    
    # 使用简单的等间距方法生成初始标准长度
    min_val = min(min_lengths)
    max_val = max(min_lengths)
    
    if num_standards <= 1:
        return [max_val]
    
    step = (max_val - min_val) / (num_standards - 1)
    initial_solution = [min_val + i * step for i in range(num_standards)]
    
    return sorted(initial_solution)


if __name__ == "__main__":
    # 示例使用
    from .ga import create_sample_data, GAConfig
    
    print("模拟退火算法钢支架长度优化示例")
    
    # 创建示例数据
    steel_supports = create_sample_data(20)
    
    # 创建初始解
    initial_solution = create_initial_solution(steel_supports, 6)
    print(f"初始解: {initial_solution}")
    
    # 配置SA参数
    sa_config = SAConfig(
        initial_temperature=1000.0,
        final_temperature=0.1,
        cooling_rate=0.95,
        max_iterations=200
    )
    
    # 运行SA优化
    sa = SimulatedAnnealing(sa_config)
    result = sa.optimize(initial_solution, steel_supports)
    
    # 输出结果
    print(f"\nSA优化结果:")
    print(f"标准长度: {result['standard_lengths']}")
    print(f"规格数量: {result['num_standards']}")
    print(f"总成本: {result['total_cost']:.2f} 元")
    print(f"总浪费: {result['total_waste']:.2f} mm")
    print(f"接受率: {result['acceptance_rate']:.3f}")
    print(f"迭代次数: {result['iterations']}")
    
    # 测试混合算法
    print("\n" + "="*50)
    print("GA-SA混合算法测试")
    
    ga_config = GAConfig(population_size=30, max_generations=50)
    hybrid = HybridGASA(ga_config, sa_config)
    hybrid_result = hybrid.optimize(steel_supports)
    
    print(f"\n混合算法结果:")
    print(f"优化方法: {hybrid_result['optimization_method']}")
    print(f"标准长度: {hybrid_result['standard_lengths']}")
    print(f"总成本: {hybrid_result['total_cost']:.2f} 元")

