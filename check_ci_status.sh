#!/bin/bash

# GitHub Actions CI状态检查脚本
# 用于验证CI配置是否正常工作

echo "🔍 GitHub Actions CI 状态检查"
echo "================================"

# 项目信息
REPO_OWNER="wangleiwps"
REPO_NAME="metro-esc-platform-ga-optimizer"
BRANCH="feature/ga-core"

echo "📋 项目信息:"
echo "  仓库: ${REPO_OWNER}/${REPO_NAME}"
echo "  分支: ${BRANCH}"
echo "  最新提交: $(git rev-parse --short HEAD)"
echo "  提交信息: $(git log -1 --pretty=format:'%s')"
echo ""

# 检查workflow文件
echo "📁 检查workflow文件:"
if [ -f ".github/workflows/python-ci.yml" ]; then
    echo "  ✅ python-ci.yml 文件存在"
    echo "  📄 文件大小: $(wc -c < .github/workflows/python-ci.yml) bytes"
    echo "  📝 文件行数: $(wc -l < .github/workflows/python-ci.yml) lines"
else
    echo "  ❌ python-ci.yml 文件不存在"
    exit 1
fi
echo ""

# 检查YAML语法
echo "🔧 检查YAML语法:"
if command -v python3 &> /dev/null; then
    python3 -c "
import yaml
import sys
try:
    with open('.github/workflows/python-ci.yml', 'r') as f:
        yaml.safe_load(f)
    print('  ✅ YAML语法正确')
except yaml.YAMLError as e:
    print(f'  ❌ YAML语法错误: {e}')
    sys.exit(1)
except Exception as e:
    print(f'  ⚠️  无法验证YAML: {e}')
"
else
    echo "  ⚠️  Python3未安装，跳过YAML语法检查"
fi
echo ""

# 检查必要的依赖文件
echo "📦 检查项目文件:"
files_to_check=("requirements.txt" "src/ga.py" "src/sa.py" "src/runner.py" "tests/test_ga.py" "tests/test_sa.py" "main.py")

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
    fi
done
echo ""

# 显示GitHub Actions链接
echo "🔗 相关链接:"
echo "  📊 Actions页面: https://github.com/${REPO_OWNER}/${REPO_NAME}/actions"
echo "  🌿 当前分支: https://github.com/${REPO_OWNER}/${REPO_NAME}/tree/${BRANCH}"
echo "  📋 Workflow文件: https://github.com/${REPO_OWNER}/${REPO_NAME}/blob/${BRANCH}/.github/workflows/python-ci.yml"
echo ""

# 显示预期的CI jobs
echo "🎯 预期的CI Jobs:"
echo "  1. test (Python 3.8, 3.9, 3.10, 3.11)"
echo "  2. integration-test"
echo "  3. code-quality"
echo "  4. summary"
echo ""

# 显示验证步骤
echo "✅ 验证步骤:"
echo "  1. 访问Actions页面查看workflow运行状态"
echo "  2. 确认所有jobs都成功运行"
echo "  3. 检查测试覆盖率报告"
echo "  4. 验证工件上传成功"
echo ""

echo "🚀 CI配置推送成功！"
echo "请访问GitHub Actions页面查看运行状态。"
echo "================================"

