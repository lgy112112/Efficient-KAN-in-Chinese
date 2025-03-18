# 清理旧构建
rm -rf build/ dist/ ikan.egg-info/

# 生成新包
python setup.py sdist bdist_wheel

# 检查 wheel 文件内容
unzip -l dist/ikan-1.2.8-py3-none-any.whl

# 检查源码包内容
tar -tvf dist/ikan-1.2.8.tar.gz

# 上传到正式 PyPI 仓库
twine upload dist/*

# 测试安装
pip install ikan==1.2.8 --force-reinstall