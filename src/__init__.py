"""
地铁疏散平台钢支架长度优化算法包

这个包实现了基于遗传算法(GA)和模拟退火(SA)的钢支架长度优化算法，
支持动态安全余量计算和成本敏感聚类。

主要模块:
- ga: 遗传算法实现
- sa: 模拟退火算法实现  
- runner: 算法运行器和结果生成

作者: wangleiwps
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "wangleiwps"
__email__ = "wangleiwps@163.com"

from .ga import (
    GeneticAlgorithm, GAConfig, SteelSupport, 
    DynamicSafetyMargin, CostSensitiveClustering,
    create_sample_data
)

from .sa import (
    SimulatedAnnealing, SAConfig, HybridGASA,
    create_initial_solution
)

from .runner import OptimizationRunner

__all__ = [
    # GA模块
    'GeneticAlgorithm', 'GAConfig', 'SteelSupport',
    'DynamicSafetyMargin', 'CostSensitiveClustering',
    'create_sample_data',
    
    # SA模块
    'SimulatedAnnealing', 'SAConfig', 'HybridGASA',
    'create_initial_solution',
    
    # 运行器
    'OptimizationRunner'
]

