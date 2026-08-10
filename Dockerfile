FROM python:3.11-slim

ENV PYTHONPATH=/app/novelAgent

WORKDIR /app/novelAgent

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget netcat-openbsd && rm -rf /var/lib/apt/lists/*

# 复制依赖文件，分层缓存加速构建
COPY requirements.txt .
# --no-update-deps 锁定全量依赖版本，适配Mac OS11旧开发环境，消除跨环境版本冲突
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --timeout 1000

# 复制全部业务源码（不含模型、向量库数据）
COPY . .

# 赋予启动脚本执行权限
RUN chmod +x ./entrypoint.sh

# 暴露FastAPI服务端口
EXPOSE 8000

# 容器入口脚本
ENTRYPOINT ["./entrypoint.sh"]
