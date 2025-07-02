#!/usr/bin/env python3
"""
钢梁长度标准化工具

将算法生成的精确长度调整为符合工程实际的标准长度
"""

import math
from typing import List, Tuple
import numpy as np

class LengthStandardizer:
    """长度标准化器"""
    
    def __init__(self, rounding_unit: int = 10):
        """
        初始化长度标准化器
        
        Args:
            rounding_unit: 长度取整单位 (mm)，默认为10mm
        """
        self.rounding_unit = rounding_unit
    
    def round_to_standard(self, length: float) -> int:
        """
        将长度调整为标准长度
        
        Args:
            length: 原始长度 (mm)
            
        Returns:
            标准化后的长度 (mm)
        """
        # 向上取整到最近的rounding_unit倍数
        return math.ceil(length / self.rounding_unit) * self.rounding_unit
    
    def standardize_lengths(self, lengths: List[float]) -> List[int]:
        """
        批量标准化长度列表
        
        Args:
            lengths: 原始长度列表
            
        Returns:
            标准化后的长度列表
        """
        standardized = [self.round_to_standard(length) for length in lengths]
        
        # 去重并排序
        unique_lengths = sorted(list(set(standardized)))
        
        return unique_lengths
    
    def optimize_standard_lengths(self, required_lengths: List[float], 
                                 max_standards: int = 10) -> Tuple[List[int], List[int]]:
        """
        优化标准长度，确保覆盖所有需求且数量合理
        
        Args:
            required_lengths: 所需长度列表
            max_standards: 最大标准规格数量
            
        Returns:
            (标准长度列表, 每种标准长度的需求数量)
        """
        # 将所有需求长度标准化
        standardized_required = [self.round_to_standard(length) for length in required_lengths]
        
        # 统计每种标准长度的需求数量
        length_counts = {}
        for length in standardized_required:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # 按需求数量排序，选择最常用的标准长度
        sorted_lengths = sorted(length_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 限制标准规格数量
        if len(sorted_lengths) > max_standards:
            # 保留需求量最大的规格
            selected_lengths = sorted_lengths[:max_standards]
            
            # 将剩余需求分配到最接近的标准长度
            selected_dict = dict(selected_lengths)
            remaining_lengths = sorted_lengths[max_standards:]
            
            for length, count in remaining_lengths:
                # 找到最接近的标准长度
                closest_length = min(selected_dict.keys(), key=lambda x: abs(x - length))
                selected_dict[closest_length] += count
            
            final_lengths = list(selected_dict.keys())
            final_quantities = list(selected_dict.values())
        else:
            final_lengths = [item[0] for item in sorted_lengths]
            final_quantities = [item[1] for item in sorted_lengths]
        
        # 排序
        sorted_pairs = sorted(zip(final_lengths, final_quantities))
        final_lengths = [pair[0] for pair in sorted_pairs]
        final_quantities = [pair[1] for pair in sorted_pairs]
        
        return final_lengths, final_quantities
    
    def validate_coverage(self, standard_lengths: List[int], 
                         required_lengths: List[float]) -> Tuple[bool, List[str]]:
        """
        验证标准长度是否能覆盖所有需求
        
        Args:
            standard_lengths: 标准长度列表
            required_lengths: 需求长度列表
            
        Returns:
            (是否完全覆盖, 问题列表)
        """
        issues = []
        all_covered = True
        
        for req_length in required_lengths:
            # 找到能满足需求的最小标准长度
            suitable_standards = [std for std in standard_lengths if std >= req_length]
            
            if not suitable_standards:
                all_covered = False
                issues.append(f"需求长度 {req_length:.1f}mm 无法被任何标准长度覆盖")
            else:
                min_suitable = min(suitable_standards)
                waste = min_suitable - req_length
                if waste > self.rounding_unit * 2:  # 浪费超过2个单位
                    issues.append(f"需求长度 {req_length:.1f}mm 使用标准长度 {min_suitable}mm，浪费 {waste:.1f}mm")
        
        return all_covered, issues

def test_length_standardizer():
    """测试长度标准化器"""
    print("🔧 测试长度标准化器")
    
    standardizer = LengthStandardizer(rounding_unit=10)
    
    # 测试单个长度标准化
    test_lengths = [1251.33, 1354.57, 1475.18, 1886.96, 2086.50, 2303.95, 2407.01]
    print(f"\n📏 原始长度: {test_lengths}")
    
    standardized = standardizer.standardize_lengths(test_lengths)
    print(f"📐 标准化后: {standardized}")
    
    # 测试优化标准长度
    required_lengths = [1200, 1350, 1470, 1880, 2080, 2300, 2400, 1250, 1360, 1480]
    print(f"\n📋 需求长度: {required_lengths}")
    
    opt_lengths, quantities = standardizer.optimize_standard_lengths(required_lengths, max_standards=5)
    print(f"🎯 优化后标准长度: {opt_lengths}")
    print(f"📊 对应数量: {quantities}")
    
    # 验证覆盖性
    covered, issues = standardizer.validate_coverage(opt_lengths, required_lengths)
    print(f"\n✅ 覆盖性验证: {'通过' if covered else '失败'}")
    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")

if __name__ == "__main__":
    test_length_standardizer()

