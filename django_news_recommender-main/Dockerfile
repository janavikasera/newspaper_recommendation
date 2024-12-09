FROM python:3.8.9
WORKDIR /build
ENV TZ=Asia/Shanghai
COPY requirements.txt /build

# 安装 NLTK 模块
RUN pip install nltk

# 下载 NLTK 数据包
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

RUN echo "正在安装依赖包..." && \
    pip install -r /build/requirements.txt -i https://mirrors.aliyun.com/pypi/simple  && \
    echo "依赖安装成功..."
EXPOSE 8000
