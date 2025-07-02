#!/usr/bin/env python3
"""
改进的遗传算法模块 - 集成长度标准化

在原有GA算法基础上，添加工程实际的长度标准化约束
确保生成的标准长度为10的倍数，符合实际生产要求
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.ga import *
from length_standardizer import LengthStandardizer
import math

class ImprovedGeneticAlgorithm(GeneticAlgorithm):
    """改进的遗传算法，集成长度标准化"""
    
    def __init__(self, config: GAConfig, rounding_unit: int = 10):
        """
        初始化改进的遗传算法
        
        Args:
            config: GA配置参数
            rounding_unit: 长度取整单位 (mm)
        """
        super().__init__(config)
        self.standardizer = LengthStandardizer(rounding_unit)
        self.rounding_unit = rounding_unit
    
    def create_individual(self, steel_supports: List[SteelSupport]) -> List[float]:
        """
        创建个体，确保标准长度为取整单位的倍数
        
        Args:
            steel_supports: 钢支架列表
            
        Returns:
            个体 (标准化的标准长度列表)
        """
        # 计算每个钢支架的最小所需长度
        min_lengths = []
        for support in steel_supports:
            margin = self.safety_margin.calculate_margin(
                measurement_precision=1.0 + support.measurement_error,
                geometric_complexity=support.geometric_complexity,
                construction_condition=support.construction_condition
            )
            min_length = support.required_length + margin
            min_lengths.append(min_length)
        
        # 将最小长度标准化为取整单位的倍数
        standardized_min_lengths = [self.standardizer.round_to_standard(length) 
                                   for length in min_lengths]
        
        # 使用聚类算法生成初始标准长度
        k = min(10, len(set(standardized_min_lengths)))
        if k > 1:
            centers, _ = self.clustering.improved_kmeans(standardized_min_lengths, k)
        else:
            centers = standardized_min_lengths
        
        # 标准化聚类中心
        standardized_centers = [self.standardizer.round_to_standard(center) 
                               for center in centers]
        
        # 去重并排序
        unique_centers = sorted(list(set(standardized_centers)))
        
        # 确保覆盖所有需求
        final_standards = self._ensure_coverage(unique_centers, standardized_min_lengths)
        
        return final_standards
    
    def _ensure_coverage(self, standards: List[int], required_lengths: List[int]) -> List[int]:
        """
        确保标准长度能覆盖所有需求长度
        
        Args:
            standards: 当前标准长度列表
            required_lengths: 需求长度列表
            
        Returns:
            调整后的标准长度列表
        """
        adjusted_standards = standards.copy()
        
        for req_length in required_lengths:
            # 检查是否有标准长度能满足需求
            suitable = [std for std in adjusted_standards if std >= req_length]
            
            if not suitable:
                # 如果没有合适的标准长度，添加一个
                new_standard = self.standardizer.round_to_standard(req_length)
                adjusted_standards.append(new_standard)
        
        # 去重并排序
        return sorted(list(set(adjusted_standards)))
    
    def mutate(self, individual: List[float]) -> List[float]:
        """
        变异操作，确保变异后的长度仍为标准长度
        
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
            
            # 生成变异值（向上或向下调整1-3个取整单位）
            direction = random.choice([-1, 1])
            steps = random.randint(1, 3)
            adjustment = direction * steps * self.rounding_unit
            
            new_value = mutated[idx] + adjustment
            # 确保最小长度
            new_value = max(self.rounding_unit, new_value)
            
            mutated[idx] = new_value
        
        return sorted(mutated)
    
    def evaluate_fitness(self, individual: List[float], steel_supports: List[SteelSupport]) -> float:
        """
        评估个体适应度，考虑标准化约束
        
        Args:
            individual: 个体 (标准长度列表)
            steel_supports: 钢支架列表
            
        Returns:
            适应度值
        """
        # 确保个体中的长度都是标准化的
        standardized_individual = [self.standardizer.round_to_standard(length) 
                                  for length in individual]
        standardized_individual = sorted(list(set(standardized_individual)))
        
        # 计算每个钢支架应该使用的标准长度
        assignments = []
        penalty = 0
        
        for support in steel_supports:
            margin = self.safety_margin.calculate_margin(
                measurement_precision=1.0 + support.measurement_error,
                geometric_complexity=support.geometric_complexity,
                construction_condition=support.construction_condition
            )
            min_required = support.required_length + margin
            min_required_std = self.standardizer.round_to_standard(min_required)
            
            # 选择满足要求的最小标准长度
            suitable_lengths = [length for length in standardized_individual 
                              if length >= min_required_std]
            
            if suitable_lengths:
                assigned_length = min(suitable_lengths)
                assignments.append(assigned_length)
                
                # 计算浪费惩罚
                waste = assigned_length - min_required
                if waste > self.rounding_unit * 3:  # 浪费超过3个单位时添加惩罚
                    penalty += waste * 0.01
            else:
                # 如果没有合适的长度，使用最大的标准长度并添加重惩罚
                if standardized_individual:
                    assignments.append(max(standardized_individual))
                    penalty += 1000  # 重惩罚
                else:
                    assignments.append(min_required_std)
                    penalty += 1000
        
        # 计算每种标准长度的使用数量
        length_counts = {}
        for length in assignments:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 计算总成本
        lengths = list(length_counts.keys())
        quantities = list(length_counts.values())
        total_cost = self.clustering.calculate_total_cost(lengths, quantities)
        
        # 添加规格数量惩罚（鼓励减少规格种类）
        spec_penalty = len(lengths) * 50
        
        # 总成本包括惩罚
        final_cost = total_cost + penalty + spec_penalty
        
        # 适应度 = 1 / (1 + 总成本)
        return 1.0 / (1.0 + final_cost)
    
    def optimize(self, steel_supports: List[SteelSupport]) -> Dict:
        """
        运行优化算法
        
        Args:
            steel_supports: 钢支架列表
            
        Returns:
            优化结果字典
        """
        print("🔧 运行改进的遗传算法（集成长度标准化）...")
        
        # 运行原始GA算法
        result = super().optimize(steel_supports)
        
        # 对最终结果进行标准化处理
        best_individual = result['standard_lengths']
        standardized_lengths = self.standardizer.standardize_lengths(best_individual)
        
        # 重新计算优化后的分配和成本
        assignments = []
        for support in steel_supports:
            margin = self.safety_margin.calculate_margin(
                measurement_precision=1.0 + support.measurement_error,
                geometric_complexity=support.geometric_complexity,
                construction_condition=support.construction_condition
            )
            min_required = support.required_length + margin
            min_required_std = self.standardizer.round_to_standard(min_required)
            
            suitable_lengths = [length for length in standardized_lengths 
                              if length >= min_required_std]
            
            if suitable_lengths:
                assignments.append(min(suitable_lengths))
            else:
                # 如果标准长度不够，添加一个新的标准长度
                new_standard = self.standardizer.round_to_standard(min_required)
                standardized_lengths.append(new_standard)
                standardized_lengths = sorted(list(set(standardized_lengths)))
                assignments.append(new_standard)
        
        # 计算最终统计
        length_counts = {}
        for length in assignments:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        lengths = list(length_counts.keys())
        quantities = list(length_counts.values())
        total_cost = self.clustering.calculate_total_cost(lengths, quantities)
        
        # 计算浪费
        total_waste = sum(assigned - support.required_length 
                         for assigned, support in zip(assignments, steel_supports))
        
        # 更新结果
        result.update({
            'standard_lengths': sorted(standardized_lengths),
            'assignments': assignments,
            'length_quantities': dict(zip(lengths, quantities)),
            'total_cost': total_cost,
            'total_waste': total_waste,
            'num_standards': len(standardized_lengths),
            'rounding_unit': self.rounding_unit
        })
        
        print(f"✅ 标准化完成！生成 {len(standardized_lengths)} 种标准规格")
        print(f"📏 标准长度: {standardized_lengths}")
        
        return result

def test_improved_ga():
    """测试改进的GA算法"""
    print("🧪 测试改进的遗传算法")
    
    # 创建测试数据
    steel_supports = create_sample_data(10)
    
    # 配置参数
    config = GAConfig(
        population_size=50,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    # 运行改进的GA算法
    improved_ga = ImprovedGeneticAlgorithm(config, rounding_unit=10)
    result = improved_ga.optimize(steel_supports)
    
    print(f"\n📊 优化结果:")
    print(f"标准长度: {result['standard_lengths']}")
    print(f"规格数量: {result['num_standards']}")
    print(f"总成本: {result['total_cost']:.2f}")
    print(f"总浪费: {result['total_waste']:.2f}mm")
    
    # 验证标准化
    for length in result['standard_lengths']:
        if length % 10 != 0:
            print(f"❌ 错误：长度 {length} 不是10的倍数")
        else:
            print(f"✅ 长度 {length} 符合标准")

if __name__ == "__main__":
    test_improved_ga()

