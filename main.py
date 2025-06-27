#!/usr/bin/env python3
"""
地铁疏散平台钢支架长度优化算法主程序

使用示例:
    python main.py --algorithm ga --input data.csv --output results/
    python main.py --algorithm sa --input data.csv --output results/
    python main.py --algorithm hybrid --input data.csv --output results/
    python main.py --compare --input data.csv --output results/
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import (
    OptimizationRunner, GAConfig, SAConfig, 
    create_sample_data
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="地铁疏散平台钢支架长度优化算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --algorithm ga --input data.csv --output results/
  %(prog)s --algorithm sa --input data.csv --output results/  
  %(prog)s --algorithm hybrid --input data.csv --output results/
  %(prog)s --compare --input data.csv --output results/
  %(prog)s --demo --output demo_results/
        """
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        choices=['ga', 'sa', 'hybrid'],
        help='选择优化算法: ga(遗传算法), sa(模拟退火), hybrid(混合算法)'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='比较所有算法性能'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='运行演示模式（使用示例数据）'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='输入CSV文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='输出目录路径 (默认: output)'
    )
    
    parser.add_argument(
        '--population-size',
        type=int,
        default=100,
        help='GA种群大小 (默认: 100)'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=300,
        help='GA最大代数 (默认: 300)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1000.0,
        help='SA初始温度 (默认: 1000.0)'
    )
    
    parser.add_argument(
        '--cooling-rate',
        type=float,
        default=0.95,
        help='SA冷却率 (默认: 0.95)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def run_single_algorithm(runner, algorithm, steel_supports, args):
    """运行单个算法"""
    print(f"运行{algorithm.upper()}算法...")
    
    if algorithm == 'ga':
        config = GAConfig(
            population_size=args.population_size,
            max_generations=args.generations
        )
        result = runner.run_ga_optimization(steel_supports, config)
        
    elif algorithm == 'sa':
        config = SAConfig(
            initial_temperature=args.temperature,
            cooling_rate=args.cooling_rate
        )
        result = runner.run_sa_optimization(steel_supports, config=config)
        
    elif algorithm == 'hybrid':
        ga_config = GAConfig(
            population_size=args.population_size,
            max_generations=args.generations
        )
        sa_config = SAConfig(
            initial_temperature=args.temperature,
            cooling_rate=args.cooling_rate
        )
        result = runner.run_hybrid_optimization(steel_supports, ga_config, sa_config)
    
    # 输出结果
    print(f"\n{algorithm.upper()}优化结果:")
    print(f"标准长度数量: {result['num_standards']}")
    print(f"总成本: {result['total_cost']:.2f} 元")
    print(f"总浪费: {result['total_waste']:.2f} mm")
    
    if args.verbose:
        print(f"标准长度: {result['standard_lengths']}")
    
    # 生成订货清单
    csv_path = runner.generate_order_csv(result)
    print(f"订货清单已保存: {csv_path}")
    
    return result


def run_comparison(runner, steel_supports, args):
    """运行算法比较"""
    print("运行算法性能比较...")
    
    comparison_result = runner.compare_algorithms(steel_supports)
    
    # 显示比较结果
    print("\n算法比较结果:")
    summary_df = comparison_result['comparison_summary']
    print(summary_df.to_string(index=False))
    
    # 生成所有订货清单
    print("\n生成订货清单...")
    for algorithm in ['GA', 'SA', 'GA-SA', 'Traditional']:
        runner.generate_order_csv(comparison_result[algorithm])
    
    # 生成对比图表
    print("\n生成对比图表...")
    plot_files = runner.generate_comparison_plots(comparison_result)
    
    for plot_name, plot_path in plot_files.items():
        print(f"{plot_name}: {plot_path}")
    
    # 生成报告
    print("\n生成优化报告...")
    report_path = runner.generate_report(comparison_result, steel_supports)
    print(f"报告已保存: {report_path}")
    
    return comparison_result


def main():
    """主函数"""
    args = parse_arguments()
    
    # 验证参数
    if not (args.algorithm or args.compare or args.demo):
        print("错误: 必须指定 --algorithm, --compare 或 --demo 参数")
        sys.exit(1)
    
    if not args.demo and not args.input:
        print("错误: 非演示模式下必须指定 --input 参数")
        sys.exit(1)
    
    # 创建运行器
    runner = OptimizationRunner(args.output)
    
    # 加载数据
    if args.demo:
        print("演示模式: 使用示例数据")
        steel_supports = create_sample_data(50)
        
        # 保存示例数据
        import pandas as pd
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
        sample_path = Path(args.output) / "demo_steel_supports.csv"
        sample_df.to_csv(sample_path, index=False)
        print(f"示例数据已保存: {sample_path}")
        
    else:
        print(f"从文件加载数据: {args.input}")
        if not os.path.exists(args.input):
            print(f"错误: 输入文件不存在: {args.input}")
            sys.exit(1)
        
        try:
            steel_supports = runner.load_steel_supports_from_csv(args.input)
            print(f"成功加载 {len(steel_supports)} 个钢支架数据")
        except Exception as e:
            print(f"错误: 加载数据失败: {e}")
            sys.exit(1)
    
    # 执行算法
    try:
        if args.compare:
            result = run_comparison(runner, steel_supports, args)
        else:
            result = run_single_algorithm(runner, args.algorithm, steel_supports, args)
        
        print(f"\n优化完成！所有结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"错误: 算法执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

