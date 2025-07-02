# 地铁疏散平台钢支架长度优化器

[![Python CI](https://github.com/wangleiwps/metro-esc-platform-ga-optimizer/actions/workflows/python-ci.yml/badge.svg?branch=feature/ga-core)](https://github.com/wangleiwps/metro-esc-platform-ga-optimizer/actions/workflows/python-ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 项目简介

本项目是《地铁疏散平台无轨化施工方法及辅助安装设备研究》SCI论文的配套代码实现，专注于钢支架长度优化问题。

### 🎯 核心创新点

- **动态安全余量计算**：基于σ×δ×ε×f四因子公式的自适应安全余量策略
- **成本敏感聚类**：改进K-means算法，综合考虑材料、标准化和浪费成本
- **GA-SA混合优化**：遗传算法与模拟退火算法的两阶段优化框架
- **长度标准化约束**：确保所有标准长度为10mm倍数，符合工程实际要求
- **智能订货清单**：自动生成长度-数量优化的CSV订货清单

## 🔧 新增功能：长度标准化

### 问题解决
- **原问题**：算法生成的标准长度包含小数（如1251.33mm），不符合钢梁生产标准
- **解决方案**：集成长度标准化器，确保所有标准长度为10的倍数
- **工程价值**：符合实际生产要求，便于采购和库存管理

### 使用方法
```bash
# 使用改进的标准化算法
python main_improved.py --demo --compare --output results/

# 处理客户采购表
python data_converter.py --input 客户采购表.csv
python main_improved.py --input 客户采购表_converted.csv --compare --output results/

# 演示标准化效果
python demo_standardized_lengths.py
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- NumPy, Pandas, Matplotlib
- 详见 `requirements.txt`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 演示模式 - 使用内置测试数据
python main.py --demo --output results/

# 算法比较模式 - 对比GA、SA、GA-SA三种算法
python main.py --demo --compare --output results/

# 使用自定义数据
python main.py --input your_data.csv --algorithm ga --output results/

# 详细输出模式
python main.py --demo --compare --output results/ --verbose
```

## 📊 算法性能

基于50个钢支架的测试数据：

| 算法 | 成本节约 | 浪费减少 | 规格减少 | 运行时间 |
|------|----------|----------|----------|----------|
| GA | 12.57% | 22.36% | 50% | ~30s |
| SA | 22.99% | 121.01% | 65% | ~25s |
| GA-SA | 13.10% | 3.03% | 50% | ~45s |

## 📁 项目结构

```
metro-esc-platform-ga-optimizer/
├── src/                    # 核心算法模块
│   ├── ga.py              # 遗传算法实现
│   ├── sa.py              # 模拟退火算法实现
│   └── runner.py          # 算法运行器和结果处理
├── tests/                 # 单元测试
│   ├── test_ga.py         # GA算法测试
│   └── test_sa.py         # SA算法测试
├── main.py                # 主程序入口
├── requirements.txt       # 项目依赖
└── .github/workflows/     # CI/CD配置
```

## 🔬 算法详解

### 动态安全余量计算

```python
safety_margin = base_margin * σ * δ * ε * f
```

- **σ**: 测量精度因子 (0.8-1.2)
- **δ**: 施工难度因子 (0.9-1.3)  
- **ε**: 几何复杂度因子 (0.85-1.15)
- **f**: 材料特性因子 (0.95-1.05)

### 成本敏感聚类

改进的K-means算法，目标函数：

```
minimize: α×材料成本 + β×标准化成本 + γ×浪费成本
```

### GA-SA混合框架

1. **阶段1 - 遗传算法**：全局搜索最优解空间
2. **阶段2 - 模拟退火**：局部精细优化
3. **结果融合**：选择最优解作为最终结果

## 📈 输出结果

### CSV订货清单
- 标准长度规格
- 每种规格的数量
- 成本统计信息
- 浪费分析报告

### 可视化图表
- 成本对比图
- 改进百分比图
- 长度分布直方图
- 优化过程曲线

### 性能报告
- JSON格式详细分析
- 算法收敛性评估
- 参数敏感性分析

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_ga.py -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

## 📚 论文支持

本代码库完全支持SCI论文实验需求：

- ✅ **可重现实验**：固定随机种子，确保结果一致性
- ✅ **性能对比**：支持多算法对比实验
- ✅ **统计分析**：提供详细的统计数据和显著性检验
- ✅ **图表生成**：自动生成论文所需的对比图表

## 🔧 开发

### 代码质量

项目使用完整的CI/CD流水线确保代码质量：

- **多版本测试**：Python 3.8-3.11
- **代码格式化**：black, isort
- **静态分析**：flake8, mypy
- **安全扫描**：bandit, safety
- **测试覆盖率**：>85%

### 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- **作者**: wangleiwps
- **邮箱**: wangleiwps@163.com
- **项目链接**: https://github.com/wangleiwps/metro-esc-platform-ga-optimizer

## 🙏 致谢

感谢所有为地铁疏散平台优化研究做出贡献的研究者和开发者。

---

**注意**: 本项目为学术研究项目，代码仅供研究和学习使用。

