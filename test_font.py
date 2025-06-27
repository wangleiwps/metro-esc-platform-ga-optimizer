#!/usr/bin/env python3
"""
测试matplotlib中文字体配置
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def test_chinese_fonts():
    """测试中文字体显示"""
    
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    algorithms = ['传统方法', 'GA算法', 'SA算法', 'GA-SA混合']
    costs = [18638, 16321, 14356, 16789]
    
    # 创建柱状图
    bars = ax.bar(algorithms, costs, color=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # 设置中文标题和标签
    ax.set_title('钢支架优化算法成本对比', fontsize=16, fontweight='bold')
    ax.set_xlabel('优化算法', fontsize=12)
    ax.set_ylabel('总成本 (元)', fontsize=12)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.0f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('/home/ubuntu/metro-esc-platform-ga-optimizer/font_test.png', 
                dpi=300, bbox_inches='tight')
    
    print("✅ 字体测试图片已保存为 font_test.png")
    print("请检查图片中的中文是否正确显示")
    
    # 显示可用字体信息
    print("\n📋 系统中可用的中文字体:")
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'Noto' in f.name]
    for font in sorted(set(chinese_fonts))[:10]:
        print(f"  - {font}")
    
    # 显示当前使用的字体
    current_font = plt.rcParams['font.sans-serif'][0]
    print(f"\n🎯 当前配置的主字体: {current_font}")
    
    plt.close()

if __name__ == "__main__":
    test_chinese_fonts()

