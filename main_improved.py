#!/usr/bin/env python3
"""
改进的主程序 - 使用标准化长度的算法

集成了长度标准化功能，确保生成的标准长度符合工程实际要求
"""

import argparse
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

# 导入改进的算法
from src.ga_improved import ImprovedGeneticAlgorithm, GAConfig, SteelSupport
from src.sa import SimulatedAnnealing, SAConfig
from src.runner import OptimizationRunner
from length_standardizer import LengthStandardizer

class ImprovedAlgorithmRunner(OptimizationRunner):
    """改进的算法运行器，使用标准化长度"""
    
    def __init__(self, output_dir: str = "results", rounding_unit: int = 10):
        """
        初始化改进的算法运行器
        
        Args:
            output_dir: 输出目录
            rounding_unit: 长度取整单位 (mm)
        """
        super().__init__(output_dir)
        self.rounding_unit = rounding_unit
        self.standardizer = LengthStandardizer(rounding_unit)
    
    def run_ga_algorithm(self, steel_supports: List[SteelSupport]) -> Dict:
        """运行改进的遗传算法"""
        print("运行改进的遗传算法...")
        
        config = GAConfig(
            population_size=100,
            max_generations=300,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_rate=0.1
        )
        
        ga = ImprovedGeneticAlgorithm(config, self.rounding_unit)
        result = ga.optimize(steel_supports)
        
        return {
            'algorithm': 'Improved_GA',
            'standard_lengths': result['standard_lengths'],
            'assignments': result['assignments'],
            'total_cost': result['total_cost'],
            'total_waste': result['total_waste'],
            'num_standards': result['num_standards'],
            'length_quantities': result['length_quantities']
        }
    
    def run_traditional_method(self, steel_supports: List[SteelSupport]) -> Dict:
        """运行传统方法，也应用标准化"""
        print("计算标准化传统方法基准...")
        
        # 计算每个钢支架的所需长度（包含安全余量）
        required_lengths = []
        for support in steel_supports:
            # 使用固定30mm安全余量
            required_length = support.required_length + 30
            # 标准化长度
            std_length = self.standardizer.round_to_standard(required_length)
            required_lengths.append(std_length)
        
        # 统计每种标准长度的数量
        length_counts = {}
        for length in required_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 计算成本和浪费
        total_cost = 0
        total_waste = 0
        
        for i, support in enumerate(steel_supports):
            assigned_length = required_lengths[i]
            cost = assigned_length * 0.15  # 单价0.15元/mm
            waste = assigned_length - support.required_length
            
            total_cost += cost
            total_waste += waste
        
        return {
            'algorithm': 'Traditional_Standardized',
            'standard_lengths': sorted(list(set(required_lengths))),
            'assignments': required_lengths,
            'total_cost': total_cost,
            'total_waste': total_waste,
            'num_standards': len(set(required_lengths)),
            'length_quantities': length_counts
        }
    
    def compare_algorithms(self, steel_supports: List[SteelSupport]) -> pd.DataFrame:
        """比较不同算法的性能"""
        print("正在运行改进算法比较...")
        
        results = []
        
        # 运行改进的GA算法
        ga_result = self.run_ga_algorithm(steel_supports)
        results.append(ga_result)
        
        # 运行标准化传统方法
        traditional_result = self.run_traditional_method(steel_supports)
        results.append(traditional_result)
        
        # 创建比较DataFrame
        comparison_data = []
        for result in results:
            num_supports = len(steel_supports)
            avg_waste = result['total_waste'] / num_supports if num_supports > 0 else 0
            cost_per_support = result['total_cost'] / num_supports if num_supports > 0 else 0
            
            comparison_data.append({
                'Algorithm': result['algorithm'],
                'Total_Cost': result['total_cost'],
                'Num_Standards': result['num_standards'],
                'Total_Waste': result['total_waste'],
                'Avg_Waste_Per_Support': avg_waste,
                'Cost_Per_Support': cost_per_support
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 计算改进百分比（以传统方法为基准）
        traditional_cost = df[df['Algorithm'] == 'Traditional_Standardized']['Total_Cost'].iloc[0]
        traditional_waste = df[df['Algorithm'] == 'Traditional_Standardized']['Total_Waste'].iloc[0]
        
        df['Cost_Improvement_%'] = ((df['Total_Cost'] - traditional_cost) / traditional_cost * 100).round(2)
        df['Waste_Reduction_%'] = ((traditional_waste - df['Total_Waste']) / traditional_waste * 100).round(2)
        
        return df
    
    def generate_order_list(self, result: Dict, algorithm_name: str) -> str:
        """生成标准化订货清单"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"order_list_{algorithm_name}_{timestamp}.csv"
        filepath = self.output_dir / "csv" / filename
        
        # 创建订货清单数据
        order_data = []
        for length, quantity in result['length_quantities'].items():
            unit_price = 0.15  # 元/mm
            subtotal = length * quantity * unit_price
            order_data.append({
                '标准长度(mm)': int(length),  # 确保显示为整数
                '订购数量': quantity,
                '单价(元/mm)': f"¥{unit_price:.2f}",
                '小计(元)': f"¥{subtotal:.2f}"
            })
        
        # 添加合计行
        total_quantity = sum(result['length_quantities'].values())
        total_cost = result['total_cost']
        order_data.append({
            '标准长度(mm)': '合计',
            '订购数量': total_quantity,
            '单价(元/mm)': '-',
            '小计(元)': f"¥{total_cost:.2f}"
        })
        
        # 保存CSV文件
        df = pd.DataFrame(order_data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        return str(filepath)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='改进的钢支架长度优化算法')
    parser.add_argument('--input', '-i', help='输入CSV文件路径')
    parser.add_argument('--output', '-o', default='improved_results', help='输出目录')
    parser.add_argument('--demo', '-d', action='store_true', help='使用演示数据')
    parser.add_argument('--rounding', '-r', type=int, default=10, help='长度取整单位(mm)')
    parser.add_argument('--compare', '-c', action='store_true', help='运行算法比较')
    
    args = parser.parse_args()
    
    # 创建改进的算法运行器
    runner = ImprovedAlgorithmRunner(args.output, args.rounding)
    
    # 加载数据
    if args.demo:
        print("演示模式: 使用示例数据")
        from src.ga import create_sample_data
        steel_supports = create_sample_data(20)
        
        # 保存演示数据
        demo_data = []
        for support in steel_supports:
            demo_data.append({
                'id': support.id,
                'required_length': support.required_length,
                'measurement_error': support.measurement_error,
                'geometric_complexity': support.geometric_complexity,
                'construction_condition': support.construction_condition
            })
        
        demo_df = pd.DataFrame(demo_data)
        demo_file = runner.output_dir / "demo_steel_supports_improved.csv"
        demo_df.to_csv(demo_file, index=False)
        print(f"示例数据已保存: {demo_file}")
        
    elif args.input:
        print(f"从文件加载数据: {args.input}")
        steel_supports = runner.load_steel_supports_from_csv(args.input)
        print(f"成功加载 {len(steel_supports)} 个钢支架数据")
    else:
        print("错误: 请指定输入文件或使用 --demo 模式")
        return
    
    if args.compare:
        # 运行算法比较
        print("运行改进算法性能比较...")
        comparison_df = runner.compare_algorithms(steel_supports)
        print("\n改进算法比较结果:")
        print(comparison_df.to_string(index=False))
        
        # 生成订货清单
        print("\n生成标准化订货清单...")
        for _, row in comparison_df.iterrows():
            algorithm = row['Algorithm']
            
            # 重新运行算法获取详细结果
            if algorithm == 'Improved_GA':
                result = runner.run_ga_algorithm(steel_supports)
            else:
                result = runner.run_traditional_method(steel_supports)
            
            order_file = runner.generate_order_list(result, algorithm)
            print(f"订货清单已保存到: {order_file}")
        
        # 生成对比图表
        print("\n生成对比图表...")
        plot_files = runner.generate_comparison_plots(comparison_df)
        for plot_type, plot_file in plot_files.items():
            print(f"{plot_type}: {plot_file}")
        
        # 生成优化报告
        print("\n生成优化报告...")
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'algorithm_comparison': comparison_df.to_dict('records'),
            'parameters': {
                'rounding_unit': args.rounding,
                'num_supports': len(steel_supports)
            },
            'plot_files': plot_files
        }
        
        report_file = runner.output_dir / "reports" / f"improved_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"报告已保存: {report_file}")
        print(f"\n优化完成！所有结果已保存到: {runner.output_dir}")
        
    else:
        # 只运行改进的GA算法
        result = runner.run_ga_algorithm(steel_supports)
        print(f"\n优化结果:")
        print(f"标准长度: {result['standard_lengths']}")
        print(f"规格数量: {result['num_standards']}")
        print(f"总成本: {result['total_cost']:.2f}元")
        print(f"总浪费: {result['total_waste']:.2f}mm")
        
        # 生成订货清单
        order_file = runner.generate_order_list(result, 'Improved_GA')
        print(f"订货清单已保存到: {order_file}")

if __name__ == "__main__":
    main()

