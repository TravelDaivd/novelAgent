"""
此补丁用于解决 sentence-transformers 5.1.2 与新版 pyarrow 的兼容性问题。
在程序入口处（main.py或app.py）最开头导入此模块即可。
"""
import pyarrow
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 应用补丁：将缺失的 PyExtensionType 指向 ExtensionType
if not hasattr(pyarrow, 'PyExtensionType'):
    pyarrow.PyExtensionType = pyarrow.ExtensionType
    logger.info("[补丁已应用] 已修复 pyarrow.PyExtensionType 缺失问题")

# 可选：验证 datasets 库能正常导入
try:
    import datasets
    logger.info(f"[补丁验证] datasets 库版本 {datasets.__version__} 导入成功")
except AttributeError as e:
    logger.error(f"[补丁警告] datasets 导入仍有可能出错: {e}")
    sys.exit(1)