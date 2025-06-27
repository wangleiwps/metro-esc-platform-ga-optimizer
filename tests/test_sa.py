"""
模拟退火算法模块测试
"""

import unittest
import math
from src.sa import (
    SAConfig, SimulatedAnnealing, HybridGASA, create_initial_solution
)
from src.ga import GAConfig, create_sample_data


class TestSAConfig(unittest.TestCase):
    """SA配置测试"""
    
    def test_sa_config_creation(self):
        """测试SA配置创建"""
        config = SAConfig(
            initial_temperature=1000.0,
            final_temperature=0.1,
            cooling_rate=0.95,
            max_iterations=500
        )
        
        self.assertEqual(config.initial_temperature, 1000.0)
        self.assertEqual(config.final_temperature, 0.1)
        self.assertEqual(config.cooling_rate, 0.95)
        self.assertEqual(config.max_iterations, 500)
    
    def test_sa_config_defaults(self):
        """测试SA配置默认值"""
        config = SAConfig()
        
        self.assertGreater(config.initial_temperature, 0)
        self.assertGreater(config.final_temperature, 0)
        self.assertLess(config.final_temperature, config.initial_temperature)
        self.assertGreater(config.cooling_rate, 0)
        self.assertLess(config.cooling_rate, 1)
        self.assertGreater(config.max_iterations, 0)


class TestSimulatedAnnealing(unittest.TestCase):
    """模拟退火算法测试"""
    
    def setUp(self):
        self.config = SAConfig(
            initial_temperature=100.0,
            final_temperature=0.1,
            cooling_rate=0.9,
            max_iterations=50,
            max_iterations_per_temp=10
        )
        self.sa = SimulatedAnnealing(self.config)
        self.steel_supports = create_sample_data(10)
        self.initial_solution = create_initial_solution(self.steel_supports, 5)
    
    def test_calculate_cost(self):
        """测试成本计算"""
        cost = self.sa.calculate_cost(self.initial_solution, self.steel_supports)
        
        # 验证成本为正数
        self.assertGreater(cost, 0)
        
        # 验证成本不是无穷大（可行解）
        self.assertNotEqual(cost, float('inf'))
    
    def test_calculate_cost_infeasible(self):
        """测试不可行解的成本计算"""
        # 创建一个明显不可行的解（长度太短）
        infeasible_solution = [100, 200]  # 很短的长度
        cost = self.sa.calculate_cost(infeasible_solution, self.steel_supports)
        
        # 不可行解应该返回无穷大成本
        self.assertEqual(cost, float('inf'))
    
    def test_generate_neighbor(self):
        """测试邻域解生成"""
        neighbor = self.sa.generate_neighbor(self.initial_solution, self.steel_supports)
        
        # 验证邻域解是列表
        self.assertIsInstance(neighbor, list)
        
        # 验证邻域解不为空
        self.assertGreater(len(neighbor), 0)
        
        # 验证邻域解元素为正数
        for length in neighbor:
            self.assertGreater(length, 0)
        
        # 验证邻域解是排序的
        self.assertEqual(neighbor, sorted(neighbor))
    
    def test_acceptance_probability(self):
        """测试接受概率计算"""
        # 测试更好的解（成本降低）
        prob_better = self.sa.acceptance_probability(100, 80, 50)
        self.assertEqual(prob_better, 1.0)
        
        # 测试更差的解（成本增加）
        prob_worse = self.sa.acceptance_probability(80, 100, 50)
        self.assertGreater(prob_worse, 0)
        self.assertLess(prob_worse, 1)
        
        # 测试温度为0的情况
        prob_zero_temp = self.sa.acceptance_probability(80, 100, 0)
        self.assertEqual(prob_zero_temp, 0.0)
    
    def test_cooling_schedule(self):
        """测试冷却调度"""
        initial_temp = 100.0
        
        # 测试第0次迭代
        temp_0 = self.sa.cooling_schedule(initial_temp, 0)
        self.assertEqual(temp_0, initial_temp)
        
        # 测试后续迭代温度递减
        temp_1 = self.sa.cooling_schedule(initial_temp, 1)
        temp_2 = self.sa.cooling_schedule(initial_temp, 2)
        
        self.assertLess(temp_1, initial_temp)
        self.assertLess(temp_2, temp_1)
        
        # 验证冷却公式
        expected_temp_1 = initial_temp * (self.config.cooling_rate ** 1)
        self.assertAlmostEqual(temp_1, expected_temp_1, places=6)
    
    def test_optimize_basic(self):
        """测试基本优化功能"""
        # 使用小规模配置进行快速测试
        small_config = SAConfig(
            initial_temperature=50.0,
            final_temperature=1.0,
            cooling_rate=0.8,
            max_iterations=20,
            max_iterations_per_temp=5
        )
        small_sa = SimulatedAnnealing(small_config)
        small_supports = create_sample_data(5)
        small_initial = create_initial_solution(small_supports, 3)
        
        result = small_sa.optimize(small_initial, small_supports)
        
        # 验证结果结构
        self.assertIn('standard_lengths', result)
        self.assertIn('assignments', result)
        self.assertIn('total_cost', result)
        self.assertIn('iterations', result)
        self.assertIn('acceptance_rate', result)
        
        # 验证结果值
        self.assertGreater(len(result['standard_lengths']), 0)
        self.assertGreater(result['total_cost'], 0)
        self.assertGreaterEqual(result['acceptance_rate'], 0)
        self.assertLessEqual(result['acceptance_rate'], 1)
        self.assertEqual(len(result['assignments']), len(small_supports))
    
    def test_optimize_convergence_history(self):
        """测试优化收敛历史记录"""
        small_config = SAConfig(
            initial_temperature=50.0,
            final_temperature=1.0,
            max_iterations=10
        )
        small_sa = SimulatedAnnealing(small_config)
        small_supports = create_sample_data(3)
        small_initial = create_initial_solution(small_supports, 2)
        
        result = small_sa.optimize(small_initial, small_supports)
        
        # 验证历史记录存在
        self.assertIn('convergence_history', result)
        history = result['convergence_history']
        
        self.assertIn('temperature', history)
        self.assertIn('cost', history)
        self.assertIn('acceptance_rate', history)
        
        # 验证历史记录长度一致
        temp_history = history['temperature']
        cost_history = history['cost']
        acceptance_history = history['acceptance_rate']
        
        self.assertEqual(len(temp_history), len(cost_history))
        self.assertEqual(len(cost_history), len(acceptance_history))
        
        # 验证温度递减
        if len(temp_history) > 1:
            for i in range(1, len(temp_history)):
                self.assertLessEqual(temp_history[i], temp_history[i-1])


class TestCreateInitialSolution(unittest.TestCase):
    """初始解生成测试"""
    
    def test_create_initial_solution_basic(self):
        """测试基本初始解生成"""
        steel_supports = create_sample_data(10)
        num_standards = 5
        
        initial_solution = create_initial_solution(steel_supports, num_standards)
        
        # 验证解的结构
        self.assertIsInstance(initial_solution, list)
        self.assertEqual(len(initial_solution), num_standards)
        
        # 验证解是排序的
        self.assertEqual(initial_solution, sorted(initial_solution))
        
        # 验证解的元素为正数
        for length in initial_solution:
            self.assertGreater(length, 0)
    
    def test_create_initial_solution_edge_cases(self):
        """测试初始解生成的边界情况"""
        steel_supports = create_sample_data(5)
        
        # 测试只有一个标准长度
        solution_1 = create_initial_solution(steel_supports, 1)
        self.assertEqual(len(solution_1), 1)
        
        # 测试标准长度数量为0
        solution_0 = create_initial_solution(steel_supports, 0)
        self.assertEqual(len(solution_0), 1)  # 应该至少返回一个长度
    
    def test_create_initial_solution_coverage(self):
        """测试初始解的覆盖性"""
        steel_supports = create_sample_data(10)
        initial_solution = create_initial_solution(steel_supports, 6)
        
        # 计算所有钢支架的最小所需长度
        from src.ga import DynamicSafetyMargin
        safety_margin = DynamicSafetyMargin()
        
        min_required_lengths = []
        for support in steel_supports:
            margin = safety_margin.calculate_margin(
                measurement_precision=1.0 + support.measurement_error,
                geometric_complexity=support.geometric_complexity,
                construction_condition=support.construction_condition
            )
            min_required = support.required_length + margin
            min_required_lengths.append(min_required)
        
        # 验证初始解能覆盖所有需求
        max_required = max(min_required_lengths)
        max_solution = max(initial_solution)
        self.assertGreaterEqual(max_solution, max_required)


class TestHybridGASA(unittest.TestCase):
    """GA-SA混合算法测试"""
    
    def test_hybrid_creation(self):
        """测试混合算法创建"""
        ga_config = GAConfig(population_size=10, max_generations=5)
        sa_config = SAConfig(initial_temperature=50.0, max_iterations=10)
        
        hybrid = HybridGASA(ga_config, sa_config)
        
        # 验证混合算法包含GA和SA组件
        self.assertIsNotNone(hybrid.ga)
        self.assertIsNotNone(hybrid.sa)
    
    def test_hybrid_optimize_basic(self):
        """测试混合算法基本优化"""
        # 使用小规模配置进行快速测试
        ga_config = GAConfig(population_size=5, max_generations=3)
        sa_config = SAConfig(
            initial_temperature=20.0,
            final_temperature=1.0,
            max_iterations=5
        )
        
        hybrid = HybridGASA(ga_config, sa_config)
        small_supports = create_sample_data(5)
        
        result = hybrid.optimize(small_supports)
        
        # 验证结果结构
        self.assertIn('optimization_method', result)
        self.assertIn('standard_lengths', result)
        self.assertIn('total_cost', result)
        
        # 验证优化方法
        self.assertIn(result['optimization_method'], ['GA', 'GA+SA'])
        
        # 验证结果值
        self.assertGreater(len(result['standard_lengths']), 0)
        self.assertGreater(result['total_cost'], 0)


if __name__ == '__main__':
    unittest.main()

