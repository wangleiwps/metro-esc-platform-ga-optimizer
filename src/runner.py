"""
算法运行器模块

整合GA和SA算法，提供统一的接口用于钢支架长度优化。
支持生成CSV订货清单和对比图表。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import json
import os
from datetime import datetime
from pathlib import Path

from .ga import GeneticAlgorithm, GAConfig, SteelSupport, create_sample_data
from .sa import SimulatedAnnealing, SAConfig, HybridGASA, create_initial_solution


class OptimizationRunner:
    """优化算法运行器"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化运行器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "csv").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
    
    def load_steel_supports_from_csv(self, csv_file: str) -> List[SteelSupport]:
        """
        从CSV文件加载钢支架数据
        
        Args:
            csv_file: CSV文件路径
            
        Returns:
            钢支架列表
        """
        df = pd.read_csv(csv_file)
        
        required_columns = ['id', 'required_length', 'measurement_error', 
                          'geometric_complexity', 'construction_condition']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必需列: {col}")
        
        supports = []
        for _, row in df.iterrows():
            support = SteelSupport(
                id=str(row['id']),
                required_length=float(row['required_length']),
                measurement_error=float(row['measurement_error']),
                geometric_complexity=float(row['geometric_complexity']),
                construction_condition=float(row['construction_condition'])
            )
            supports.append(support)
        
        return supports
    
    def run_ga_optimization(self, steel_supports: List[SteelSupport], 
                           config: Optional[GAConfig] = None) -> Dict:
        """
        运行遗传算法优化
        
        Args:
            steel_supports: 钢支架列表
            config: GA配置，如果为None则使用默认配置
            
        Returns:
            优化结果
        """
        if config is None:
            config = GAConfig(
                population_size=100,
                max_generations=300,
                crossover_rate=0.8,
                mutation_rate=0.1
            )
        
        ga = GeneticAlgorithm(config)
        result = ga.optimize(steel_supports)
        result['algorithm'] = 'GA'
        result['config'] = config.__dict__
        
        return result
    
    def run_sa_optimization(self, steel_supports: List[SteelSupport],
                           initial_solution: Optional[List[float]] = None,
                           config: Optional[SAConfig] = None) -> Dict:
        """
        运行模拟退火优化
        
        Args:
            steel_supports: 钢支架列表
            initial_solution: 初始解，如果为None则自动生成
            config: SA配置，如果为None则使用默认配置
            
        Returns:
            优化结果
        """
        if config is None:
            config = SAConfig(
                initial_temperature=1000.0,
                final_temperature=0.1,
                cooling_rate=0.95,
                max_iterations=500
            )
        
        if initial_solution is None:
            initial_solution = create_initial_solution(steel_supports, 8)
        
        sa = SimulatedAnnealing(config)
        result = sa.optimize(initial_solution, steel_supports)
        result['algorithm'] = 'SA'
        result['config'] = config.__dict__
        result['initial_solution'] = initial_solution
        
        return result
    
    def run_hybrid_optimization(self, steel_supports: List[SteelSupport],
                               ga_config: Optional[GAConfig] = None,
                               sa_config: Optional[SAConfig] = None) -> Dict:
        """
        运行GA-SA混合优化
        
        Args:
            steel_supports: 钢支架列表
            ga_config: GA配置
            sa_config: SA配置
            
        Returns:
            优化结果
        """
        if ga_config is None:
            ga_config = GAConfig(population_size=80, max_generations=200)
        
        if sa_config is None:
            sa_config = SAConfig(initial_temperature=500.0, max_iterations=300)
        
        hybrid = HybridGASA(ga_config, sa_config)
        result = hybrid.optimize(steel_supports)
        result['ga_config'] = ga_config.__dict__
        result['sa_config'] = sa_config.__dict__
        
        return result
    
    def compare_algorithms(self, steel_supports: List[SteelSupport]) -> Dict:
        """
        比较不同算法的性能
        
        Args:
            steel_supports: 钢支架列表
            
        Returns:
            比较结果
        """
        print("正在运行算法比较...")
        
        # 运行GA
        print("运行遗传算法...")
        ga_result = self.run_ga_optimization(steel_supports)
        
        # 运行SA
        print("运行模拟退火算法...")
        sa_result = self.run_sa_optimization(steel_supports)
        
        # 运行混合算法
        print("运行GA-SA混合算法...")
        hybrid_result = self.run_hybrid_optimization(steel_supports)
        
        # 创建传统方法基准 (固定30mm安全余量)
        print("计算传统方法基准...")
        traditional_result = self._calculate_traditional_method(steel_supports)
        
        comparison = {
            'GA': ga_result,
            'SA': sa_result,
            'GA-SA': hybrid_result,
            'Traditional': traditional_result,
            'comparison_summary': self._create_comparison_summary([
                ('GA', ga_result),
                ('SA', sa_result), 
                ('GA-SA', hybrid_result),
                ('Traditional', traditional_result)
            ])
        }
        
        return comparison
    
    def _calculate_traditional_method(self, steel_supports: List[SteelSupport]) -> Dict:
        """
        计算传统方法 (固定30mm安全余量)
        
        Args:
            steel_supports: 钢支架列表
            
        Returns:
            传统方法结果
        """
        # 固定30mm安全余量
        required_lengths = [s.required_length + 30 for s in steel_supports]
        
        # 简单聚类：按100mm间隔分组
        min_length = min(required_lengths)
        max_length = max(required_lengths)
        
        # 生成标准长度 (100mm间隔)
        standard_lengths = []
        current = math.ceil(min_length / 100) * 100  # 向上取整到100的倍数
        while current <= max_length + 100:
            standard_lengths.append(current)
            current += 100
        
        # 分配钢支架到标准长度
        assignments = []
        for i, support in enumerate(steel_supports):
            required = required_lengths[i]
            assigned = min([length for length in standard_lengths if length >= required])
            
            assignments.append({
                'support_id': support.id,
                'required_length': support.required_length,
                'safety_margin': 30.0,
                'assigned_length': assigned,
                'waste': assigned - support.required_length
            })
        
        # 统计使用情况
        length_counts = {}
        for assignment in assignments:
            length = assignment['assigned_length']
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 计算成本 (简化)
        material_cost = sum(l * q * 0.15 for l, q in length_counts.items())
        standardization_cost = len(standard_lengths) * 50
        waste_cost = sum(a['waste'] for a in assignments) * 0.15
        total_cost = material_cost + standardization_cost + waste_cost
        
        return {
            'algorithm': 'Traditional',
            'standard_lengths': standard_lengths,
            'assignments': assignments,
            'length_quantities': length_counts,
            'total_cost': total_cost,
            'num_standards': len(standard_lengths),
            'total_waste': sum(a['waste'] for a in assignments)
        }
    
    def _create_comparison_summary(self, results: List[Tuple[str, Dict]]) -> pd.DataFrame:
        """
        创建算法比较摘要
        
        Args:
            results: (算法名称, 结果) 元组列表
            
        Returns:
            比较摘要DataFrame
        """
        summary_data = []
        
        for name, result in results:
            summary_data.append({
                'Algorithm': name,
                'Total_Cost': result['total_cost'],
                'Num_Standards': result['num_standards'],
                'Total_Waste': result['total_waste'],
                'Avg_Waste_Per_Support': result['total_waste'] / len(result['assignments']),
                'Cost_Per_Support': result['total_cost'] / len(result['assignments'])
            })
        
        df = pd.DataFrame(summary_data)
        
        # 计算改进百分比 (相对于传统方法)
        traditional_cost = df[df['Algorithm'] == 'Traditional']['Total_Cost'].iloc[0]
        traditional_waste = df[df['Algorithm'] == 'Traditional']['Total_Waste'].iloc[0]
        
        df['Cost_Improvement_%'] = ((traditional_cost - df['Total_Cost']) / traditional_cost * 100).round(2)
        df['Waste_Reduction_%'] = ((traditional_waste - df['Total_Waste']) / traditional_waste * 100).round(2)
        
        return df
    
    def generate_order_csv(self, result: Dict, filename: Optional[str] = None) -> str:
        """
        生成订货清单CSV文件
        
        Args:
            result: 优化结果
            filename: 文件名，如果为None则自动生成
            
        Returns:
            生成的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"order_list_{result['algorithm']}_{timestamp}.csv"
        
        filepath = self.output_dir / "csv" / filename
        
        # 创建订货清单数据
        order_data = []
        for length, quantity in result['length_quantities'].items():
            order_data.append({
                '标准长度(mm)': length,
                '订购数量': quantity,
                '单价(元/mm)': 0.15,
                '小计(元)': length * quantity * 0.15
            })
        
        # 按长度排序
        order_data.sort(key=lambda x: x['标准长度(mm)'])
        
        # 添加汇总行
        total_quantity = sum(item['订购数量'] for item in order_data)
        total_cost = sum(item['小计(元)'] for item in order_data)
        
        order_data.append({
            '标准长度(mm)': '合计',
            '订购数量': total_quantity,
            '单价(元/mm)': '-',
            '小计(元)': total_cost
        })
        
        # 保存CSV
        df = pd.DataFrame(order_data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"订货清单已保存到: {filepath}")
        return str(filepath)
    
    def generate_comparison_plots(self, comparison_result: Dict, 
                                save_plots: bool = True) -> Dict[str, str]:
        """
        生成对比图表
        
        Args:
            comparison_result: 算法比较结果
            save_plots: 是否保存图表
            
        Returns:
            生成的图表文件路径字典
        """
        plt.style.use('seaborn-v0_8')
        plot_files = {}
        
        # 1. 成本对比柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        summary_df = comparison_result['comparison_summary']
        
        bars = ax.bar(summary_df['Algorithm'], summary_df['Total_Cost'], 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        ax.set_title('算法成本对比', fontsize=16, fontweight='bold')
        ax.set_xlabel('算法', fontsize=12)
        ax.set_ylabel('总成本 (元)', fontsize=12)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            cost_plot_path = self.output_dir / "plots" / "cost_comparison.png"
            plt.savefig(cost_plot_path, dpi=300, bbox_inches='tight')
            plot_files['cost_comparison'] = str(cost_plot_path)
        
        plt.show()
        
        # 2. 浪费对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(summary_df['Algorithm'], summary_df['Total_Waste'],
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        ax.set_title('算法材料浪费对比', fontsize=16, fontweight='bold')
        ax.set_xlabel('算法', fontsize=12)
        ax.set_ylabel('总浪费 (mm)', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            waste_plot_path = self.output_dir / "plots" / "waste_comparison.png"
            plt.savefig(waste_plot_path, dpi=300, bbox_inches='tight')
            plot_files['waste_comparison'] = str(waste_plot_path)
        
        plt.show()
        
        # 3. 规格数量对比
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(summary_df['Algorithm'], summary_df['Num_Standards'],
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        ax.set_title('标准规格数量对比', fontsize=16, fontweight='bold')
        ax.set_xlabel('算法', fontsize=12)
        ax.set_ylabel('规格数量', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            standards_plot_path = self.output_dir / "plots" / "standards_comparison.png"
            plt.savefig(standards_plot_path, dpi=300, bbox_inches='tight')
            plot_files['standards_comparison'] = str(standards_plot_path)
        
        plt.show()
        
        # 4. 改进百分比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 成本改进
        improvement_data = summary_df[summary_df['Algorithm'] != 'Traditional']
        bars1 = ax1.bar(improvement_data['Algorithm'], improvement_data['Cost_Improvement_%'],
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax1.set_title('成本改进百分比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('算法', fontsize=12)
        ax1.set_ylabel('成本改进 (%)', fontsize=12)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 浪费减少
        bars2 = ax2.bar(improvement_data['Algorithm'], improvement_data['Waste_Reduction_%'],
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax2.set_title('浪费减少百分比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('算法', fontsize=12)
        ax2.set_ylabel('浪费减少 (%)', fontsize=12)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_plots:
            improvement_plot_path = self.output_dir / "plots" / "improvement_comparison.png"
            plt.savefig(improvement_plot_path, dpi=300, bbox_inches='tight')
            plot_files['improvement_comparison'] = str(improvement_plot_path)
        
        plt.show()
        
        return plot_files
    
    def generate_report(self, comparison_result: Dict, 
                       steel_supports: List[SteelSupport]) -> str:
        """
        生成优化报告
        
        Args:
            comparison_result: 算法比较结果
            steel_supports: 钢支架列表
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / "reports" / f"optimization_report_{timestamp}.json"
        
        # 创建报告数据
        report_data = {
            'timestamp': timestamp,
            'input_data': {
                'num_supports': len(steel_supports),
                'length_range': {
                    'min': min(s.required_length for s in steel_supports),
                    'max': max(s.required_length for s in steel_supports),
                    'avg': sum(s.required_length for s in steel_supports) / len(steel_supports)
                }
            },
            'results': comparison_result,
            'summary': comparison_result['comparison_summary'].to_dict('records')
        }
        
        # 保存JSON报告
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"优化报告已保存到: {report_path}")
        return str(report_path)


def main():
    """主函数 - 示例使用"""
    print("钢支架长度优化算法运行器")
    print("=" * 50)
    
    # 创建运行器
    runner = OptimizationRunner("output")
    
    # 创建示例数据
    print("生成示例数据...")
    steel_supports = create_sample_data(50)
    
    # 保存示例数据到CSV
    sample_data = []
    for support in steel_supports:
        sample_data.append({
            'id': support.id,
            'required_length': support.required_length,
            'measurement_error': support.measurement_error,
            'geometric_complexity': support.geometric_complexity,
            'construction_condition': support.construction_condition
        })
    
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = runner.output_dir / "sample_steel_supports.csv"
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"示例数据已保存到: {sample_csv_path}")
    
    # 运行算法比较
    print("\n开始算法比较...")
    comparison_result = runner.compare_algorithms(steel_supports)
    
    # 显示比较结果
    print("\n算法比较结果:")
    print(comparison_result['comparison_summary'])
    
    # 生成订货清单
    print("\n生成订货清单...")
    for algorithm in ['GA', 'SA', 'GA-SA', 'Traditional']:
        runner.generate_order_csv(comparison_result[algorithm])
    
    # 生成对比图表
    print("\n生成对比图表...")
    plot_files = runner.generate_comparison_plots(comparison_result)
    
    # 生成报告
    print("\n生成优化报告...")
    report_path = runner.generate_report(comparison_result, steel_supports)
    
    print("\n" + "=" * 50)
    print("优化完成！所有结果已保存到 output/ 目录")


if __name__ == "__main__":
    import math
    main()

