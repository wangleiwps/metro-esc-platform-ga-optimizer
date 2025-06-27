"""
遗传算法模块测试
"""

import unittest
import numpy as np
from src.ga import (
    GAConfig, SteelSupport, DynamicSafetyMargin, 
    CostSensitiveClustering, GeneticAlgorithm, create_sample_data
)


class TestDynamicSafetyMargin(unittest.TestCase):
    """动态安全余量测试"""
    
    def setUp(self):
        self.margin_calculator = DynamicSafetyMargin()
    
    def test_calculate_margin_normal(self):
        """测试正常情况下的安全余量计算"""
        margin = self.margin_calculator.calculate_margin(
            measurement_precision=1.0,
            geometric_complexity=1.0,
            construction_condition=1.0,
            safety_factor=1.2
        )
        expected = 30.0 * 1.0 * 1.0 * 1.0 * 1.2
        self.assertAlmostEqual(margin, expected, places=2)
    
    def test_calculate_margin_bounds(self):
        """测试安全余量的边界限制"""
        # 测试下界
        margin_low = self.margin_calculator.calculate_margin(
            measurement_precision=0.1,
            geometric_complexity=0.1,
            construction_condition=0.1,
            safety_factor=0.1
        )
        self.assertGreaterEqual(margin_low, 10.0)
        
        # 测试上界
        margin_high = self.margin_calculator.calculate_margin(
            measurement_precision=5.0,
            geometric_complexity=5.0,
            construction_condition=5.0,
            safety_factor=5.0
        )
        self.assertLessEqual(margin_high, 100.0)
    
    def test_calculate_margin_different_factors(self):
        """测试不同因子的影响"""
        base_margin = self.margin_calculator.calculate_margin(1.0, 1.0, 1.0, 1.0)
        
        # 测量精度因子影响
        high_precision_margin = self.margin_calculator.calculate_margin(2.0, 1.0, 1.0, 1.0)
        self.assertGreater(high_precision_margin, base_margin)
        
        # 几何复杂度因子影响
        high_complexity_margin = self.margin_calculator.calculate_margin(1.0, 1.5, 1.0, 1.0)
        self.assertGreater(high_complexity_margin, base_margin)


class TestCostSensitiveClustering(unittest.TestCase):
    """成本敏感聚类测试"""
    
    def setUp(self):
        self.clustering = CostSensitiveClustering()
    
    def test_calculate_total_cost(self):
        """测试总成本计算"""
        lengths = [1000, 1500, 2000]
        quantities = [10, 15, 8]
        
        cost = self.clustering.calculate_total_cost(lengths, quantities)
        
        # 验证成本为正数
        self.assertGreater(cost, 0)
        
        # 验证成本包含材料成本、标准化成本和浪费成本
        material_cost = sum(l * q * 0.15 for l, q in zip(lengths, quantities))
        self.assertGreater(cost, material_cost)
    
    def test_improved_kmeans_basic(self):
        """测试改进K-means基本功能"""
        data = [1000, 1100, 1200, 1800, 1900, 2000]
        k = 2
        
        centers, labels = self.clustering.improved_kmeans(data, k)
        
        # 验证聚类中心数量
        self.assertEqual(len(centers), k)
        
        # 验证标签数量
        self.assertEqual(len(labels), len(data))
        
        # 验证标签值在有效范围内
        for label in labels:
            self.assertIn(label, range(k))
    
    def test_improved_kmeans_edge_cases(self):
        """测试K-means边界情况"""
        # 数据点少于聚类数
        data = [1000, 1500]
        k = 5
        
        centers, labels = self.clustering.improved_kmeans(data, k)
        self.assertEqual(len(centers), len(data))
        self.assertEqual(len(labels), len(data))


class TestGeneticAlgorithm(unittest.TestCase):
    """遗传算法测试"""
    
    def setUp(self):
        self.config = GAConfig(
            population_size=20,
            max_generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        self.ga = GeneticAlgorithm(self.config)
        self.steel_supports = create_sample_data(10)
    
    def test_create_individual(self):
        """测试个体创建"""
        individual = self.ga.create_individual(self.steel_supports)
        
        # 验证个体是列表
        self.assertIsInstance(individual, list)
        
        # 验证个体不为空
        self.assertGreater(len(individual), 0)
        
        # 验证个体元素为数值
        for gene in individual:
            self.assertIsInstance(gene, (int, float))
            self.assertGreater(gene, 0)
    
    def test_evaluate_fitness(self):
        """测试适应度评估"""
        individual = self.ga.create_individual(self.steel_supports)
        fitness = self.ga.evaluate_fitness(individual, self.steel_supports)
        
        # 验证适应度为正数
        self.assertGreater(fitness, 0)
        
        # 验证适应度在合理范围内
        self.assertLessEqual(fitness, 1.0)
    
    def test_crossover(self):
        """测试交叉操作"""
        parent1 = [1000, 1500, 2000]
        parent2 = [1200, 1600, 1800]
        
        child1, child2 = self.ga.crossover(parent1, parent2)
        
        # 验证子代是列表
        self.assertIsInstance(child1, list)
        self.assertIsInstance(child2, list)
        
        # 验证子代长度
        self.assertGreater(len(child1), 0)
        self.assertGreater(len(child2), 0)
    
    def test_mutate(self):
        """测试变异操作"""
        individual = [1000, 1500, 2000]
        mutated = self.ga.mutate(individual)
        
        # 验证变异后仍是列表
        self.assertIsInstance(mutated, list)
        
        # 验证长度不变
        self.assertEqual(len(mutated), len(individual))
        
        # 验证元素为正数
        for gene in mutated:
            self.assertGreater(gene, 0)
    
    def test_tournament_selection(self):
        """测试锦标赛选择"""
        population = [[1000, 1500], [1200, 1600], [1100, 1400]]
        fitness_scores = [0.8, 0.9, 0.7]
        
        selected = self.ga.tournament_selection(population, fitness_scores)
        
        # 验证选中的个体在种群中
        self.assertIn(selected, population)
    
    def test_optimize_basic(self):
        """测试基本优化功能"""
        # 使用小规模配置进行快速测试
        small_config = GAConfig(
            population_size=10,
            max_generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        small_ga = GeneticAlgorithm(small_config)
        small_supports = create_sample_data(5)
        
        result = small_ga.optimize(small_supports)
        
        # 验证结果结构
        self.assertIn('standard_lengths', result)
        self.assertIn('assignments', result)
        self.assertIn('total_cost', result)
        self.assertIn('fitness', result)
        
        # 验证结果值
        self.assertGreater(len(result['standard_lengths']), 0)
        self.assertGreater(result['total_cost'], 0)
        self.assertGreater(result['fitness'], 0)
        self.assertEqual(len(result['assignments']), len(small_supports))


class TestSteelSupport(unittest.TestCase):
    """钢支架数据结构测试"""
    
    def test_steel_support_creation(self):
        """测试钢支架创建"""
        support = SteelSupport(
            id="S001",
            required_length=1500.0,
            measurement_error=0.1,
            geometric_complexity=1.2,
            construction_condition=1.0
        )
        
        self.assertEqual(support.id, "S001")
        self.assertEqual(support.required_length, 1500.0)
        self.assertEqual(support.measurement_error, 0.1)
        self.assertEqual(support.geometric_complexity, 1.2)
        self.assertEqual(support.construction_condition, 1.0)


class TestSampleDataGeneration(unittest.TestCase):
    """示例数据生成测试"""
    
    def test_create_sample_data(self):
        """测试示例数据创建"""
        num_supports = 20
        supports = create_sample_data(num_supports)
        
        # 验证数量
        self.assertEqual(len(supports), num_supports)
        
        # 验证每个支架的属性
        for support in supports:
            self.assertIsInstance(support, SteelSupport)
            self.assertIsInstance(support.id, str)
            self.assertGreater(support.required_length, 0)
            self.assertGreaterEqual(support.measurement_error, 0)
            self.assertGreater(support.geometric_complexity, 0)
            self.assertGreater(support.construction_condition, 0)


if __name__ == '__main__':
    unittest.main()

