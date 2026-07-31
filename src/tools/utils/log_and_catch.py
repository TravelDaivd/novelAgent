import functools
import time
import logging

logger = logging.getLogger(__name__)


def log_and_catch(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func_name = func.__name__

        # 只记录入参（跳过 self）
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


