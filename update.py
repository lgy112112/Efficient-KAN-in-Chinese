import os
import shutil
import subprocess
import zipfile
import tarfile

def main():
    # 清理旧构建
    shutil.rmtree('build/', ignore_errors=True)
    shutil.rmtree('dist/', ignore_errors=True)
    shutil.rmtree('ikan.egg-info/', ignore_errors=True)

    # 生成新包
    subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"], check=True)

    # 检查 wheel 文件内容
    with zipfile.ZipFile('dist/ikan-1.3.0-py3-none-any.whl', 'r') as zip_ref:
        zip_ref.printdir()  # 打印文件列表

    # 检查源码包内容
    with tarfile.open('dist/ikan-1.3.0.tar.gz', 'r:gz') as tar_ref:
        tar_ref.list()  # 打印文件列表

    # 上传到正式 PyPI 仓库
    subprocess.run(["twine", "upload", "dist/*"], check=True)

    # 测试安装
    # subprocess.run(["pip", "install", "ikan==1.3.0", "--force-reinstall"], check=True)

if __name__ == "__main__":
    main()