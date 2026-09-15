import functools
import time
import logging
import sys


def log_and_catch(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 获取 logger
        logger = logging.getLogger(func.__module__)

        # 如果没有 handler，自动添加
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            # 防止向上传播导致重复
            logger.propagate = False

        start = time.time()
        func_name = func.__name__

        args_str = [str(a) for a in args[1:]] if args else []
        kwargs_str = [f"{k}={v}" for k, v in kwargs.items()]
        logger.info(f"[调用] {func_name} | 入参: {', '.join(args_str + kwargs_str)}")

        try:
            result = func(*args, **kwargs)
            elapsed = round((time.time() - start) * 1000, 2)
            logger.info(f"[返回] {func_name} | 耗时 {elapsed}ms")
            return result
        except Exception as e:
            elapsed = round((time.time() - start) * 1000, 2)
            logger.error(f"[异常] {func_name} | 错误: {e} | 耗时 {elapsed}ms", exc_info=True)
            raise

    return wrapper