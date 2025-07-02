#!/usr/bin/env python3
"""
客户采购表数据转换工具

将客户提供的钢支架采购表转换为算法可处理的标准格式
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import argparse
import os

class CustomerDataConverter:
    """客户数据转换器"""
    
    def __init__(self):
        """初始化转换器"""
        # 施工难度映射
        self.difficulty_mapping = {
            '简单': 0.9,
            '中等': 1.1, 
            '困难': 1.3,
            '极难': 1.5
        }
        
        # 测量误差估算（基于施工难度）
        self.error_mapping = {
            '简单': 0.05,   # 5%误差
            '中等': 0.10,   # 10%误差
            '困难': 0.15,   # 15%误差
            '极难': 0.20    # 20%误差
        }
    
    def convert_customer_data(self, input_file: str, output_file: str = None) -> str:
        """
        转换客户采购表为标准格式
        
        Args:
            input_file: 客户采购表文件路径
            output_file: 输出文件路径（可选）
            
        Returns:
            转换后的文件路径
        """
        print(f"📋 正在读取客户采购表: {input_file}")
        
        # 读取客户数据
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding='gbk')
        
        print(f"✅ 成功读取 {len(df)} 条钢支架记录")
        
        # 显示原始数据样本
        print("\n📊 原始数据样本:")
        print(df.head())
        
        # 数据转换
        converted_data = []
        
        for idx, row in df.iterrows():
            # 提取基本信息
            support_id = row.iloc[0]  # 第一列：钢支架编号
            required_length = float(row.iloc[1])  # 第二列：实测长度
            
            # 处理施工难度（第三列或第四列）
            difficulty = self._extract_difficulty(row)
            
            # 计算参数
            measurement_error = self.error_mapping.get(difficulty, 0.10)
            geometric_complexity = np.random.uniform(0.9, 1.4)  # 几何复杂度
            construction_condition = self.difficulty_mapping.get(difficulty, 1.1)
            
            converted_data.append({
                'id': support_id,
                'required_length': required_length,
                'measurement_error': measurement_error,
                'geometric_complexity': geometric_complexity,
                'construction_condition': construction_condition
            })
        
        # 创建转换后的DataFrame
        converted_df = pd.DataFrame(converted_data)
        
        # 确定输出文件路径
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{base_name}_converted.csv"
        
        # 保存转换后的数据
        converted_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✅ 数据转换完成！")
        print(f"📁 转换后文件: {output_file}")
        print(f"📊 转换后数据样本:")
        print(converted_df.head())
        
        # 显示统计信息
        self._show_statistics(converted_df)
        
        return output_file
    
    def _extract_difficulty(self, row) -> str:
        """提取施工难度信息"""
        # 尝试从不同列提取施工难度
        for col_idx in range(2, min(len(row), 6)):  # 检查第3-6列
            value = str(row.iloc[col_idx]).strip()
            if value in self.difficulty_mapping:
                return value
        
        # 如果没有找到，根据关键词判断
        row_text = ' '.join([str(x) for x in row]).lower()
        if '困难' in row_text or '复杂' in row_text:
            return '困难'
        elif '简单' in row_text or '直线' in row_text:
            return '简单'
        else:
            return '中等'
    
    def _show_statistics(self, df: pd.DataFrame):
        """显示数据统计信息"""
        print(f"\n📈 数据统计信息:")
        print(f"  钢支架总数: {len(df)}")
        print(f"  长度范围: {df['required_length'].min():.1f} - {df['required_length'].max():.1f} mm")
        print(f"  平均长度: {df['required_length'].mean():.1f} mm")
        print(f"  测量误差范围: {df['measurement_error'].min():.1%} - {df['measurement_error'].max():.1%}")
        print(f"  施工条件因子范围: {df['construction_condition'].min():.2f} - {df['construction_condition'].max():.2f}")

def create_template():
    """创建客户数据模板"""
    template_data = {
        '钢支架编号': ['ZJ001', 'ZJ002', 'ZJ003', '...'],
        '实测长度(mm)': [2350, 1420, 2680, '...'],
        '安装位置': ['站台A区-1号位', '站台A区-2号位', '站台B区-1号位', '...'],
        '施工难度': ['中等', '简单', '困难', '...'],
        '备注': ['标准安装', '直线段', '弯道处', '...']
    }
    
    template_df = pd.DataFrame(template_data)
    template_file = 'customer_data_template.csv'
    template_df.to_csv(template_file, index=False, encoding='utf-8')
    
    print(f"📋 客户数据模板已创建: {template_file}")
    print("📝 模板格式说明:")
    print("  - 第1列: 钢支架编号（必需）")
    print("  - 第2列: 实测长度，单位mm（必需）")
    print("  - 第3列: 安装位置（可选）")
    print("  - 第4列: 施工难度（简单/中等/困难/极难）（可选）")
    print("  - 第5列: 备注信息（可选）")
    
    return template_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='客户采购表数据转换工具')
    parser.add_argument('--input', '-i', help='输入的客户采购表文件')
    parser.add_argument('--output', '-o', help='输出文件路径（可选）')
    parser.add_argument('--template', '-t', action='store_true', help='创建客户数据模板')
    parser.add_argument('--demo', '-d', action='store_true', help='使用示例数据进行演示')
    
    args = parser.parse_args()
    
    converter = CustomerDataConverter()
    
    if args.template:
        create_template()
        return
    
    if args.demo:
        # 使用示例数据
        input_file = 'customer_data_example.csv'
        print("🎯 演示模式：使用示例客户采购表")
    elif args.input:
        input_file = args.input
    else:
        print("❌ 请指定输入文件或使用 --demo 模式")
        print("💡 使用 --template 创建数据模板")
        print("💡 使用 --demo 查看演示效果")
        return
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    # 执行转换
    output_file = converter.convert_customer_data(input_file, args.output)
    
    print(f"\n🎯 下一步操作:")
    print(f"   python main.py --input {output_file} --compare --output customer_results/")

if __name__ == "__main__":
    main()

