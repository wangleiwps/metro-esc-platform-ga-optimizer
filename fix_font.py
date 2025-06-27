#!/usr/bin/env python3
"""
强制配置matplotlib中文字体
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import os

def setup_chinese_fonts():
    """强制设置中文字体"""
    
    # 字体文件路径
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
    ]
    
    # 添加字体到matplotlib
    for font_path in font_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            print(f"✅ 已添加字体: {font_path}")
    
    # 设置默认字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建字体属性对象
    chinese_font = FontProperties()
    
    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
    print(f"📋 可用的CJK字体: {available_fonts[:5]}")
    
    if available_fonts:
        chinese_font.set_family(available_fonts[0])
        print(f"🎯 使用字体: {available_fonts[0]}")
        return chinese_font
    
    return None

def create_test_plot_with_font():
    """使用指定字体创建测试图表"""
    
    # 设置字体
    chinese_font = setup_chinese_fonts()
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = ['传统方法', 'GA算法', 'SA算法', 'GA-SA混合']
    costs = [18638, 16321, 14356, 16789]
    
    bars = ax.bar(algorithms, costs, color=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # 使用指定字体设置标题和标签
    if chinese_font:
        ax.set_title('钢支架优化算法成本对比', fontproperties=chinese_font, fontsize=16, fontweight='bold')
        ax.set_xlabel('优化算法', fontproperties=chinese_font, fontsize=12)
        ax.set_ylabel('总成本 (元)', fontproperties=chinese_font, fontsize=12)
        
        # 设置x轴标签字体
        ax.set_xticklabels(algorithms, fontproperties=chinese_font)
    else:
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
    plt.savefig('/home/ubuntu/metro-esc-platform-ga-optimizer/font_test_fixed.png', 
                dpi=300, bbox_inches='tight')
    
    print("✅ 修复后的字体测试图片已保存为 font_test_fixed.png")
    plt.close()

if __name__ == "__main__":
    create_test_plot_with_font()

