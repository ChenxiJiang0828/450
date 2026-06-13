# 设置的docker基础镜像
#FROM docker.v2.aispeech.com/aispeech/aiym-intent-detection-base-image:sentence_transformer
FROM docker.v2.aispeech.com/aispeech/tsmserver:v0.0.1
# 设置镜像的基本信息
LABEL maintainer="jianming.gong@aispeech.com" description="Service For Thing Specification Model"

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /opt/aispeech

#RUN cp -r /disk2/data/users/jianming.gong/workspace/sln_develop/llm_function_call_tsl_deploy/scripts              /opt/aispeech/scripts
#RUN cp -r /disk2/data/users/jianming.gong/workspace/sln_develop/llm_function_call_tsl_deploy/src                  /opt/aispeech/tsmrt
#RUN cp -r /disk2/data/users/jianming.gong/workspace/sln_develop/llm_function_call_tsl_deploy/data                 /opt/aispeech/data
#RUN cp -r /disk2/data/users/jianming.gong/workspace/sln_develop/llm_function_call_tsl_deploy/embedding_res        /opt/aispeech/embedding_res
#RUN cp -r /disk2/data/users/jianming.gong/workspace/sln_develop/llm_function_call_tsl_deploy/function_call_data   /opt/aispeech/function_call_data

COPY scripts              /opt/aispeech/scripts
COPY tsmrt                /opt/aispeech/tsmrt
COPY embedding_res        /opt/aispeech/embedding_res
COPY function_call_data   /opt/aispeech/function_call_data
COPY vocab_data           /opt/aispeech/vocab_data

#COPY 3rdparty             /opt/aispeech/3rdparty
#COPY logs                 /opt/aispeech/logs
COPY serve                /opt/aispeech/serve
COPY requirements.txt     /opt/aispeech/requirements.txt
COPY run.sh               /opt/aispeech/run.sh
RUN  chmod +x             /opt/aispeech/run.sh

RUN pip3 install -r /opt/aispeech/requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# 设置暴露的端口
# Create workspace and Set Time zone
# 创建docker内部的项目目录, 设置工作目录为创建的目录

# Setup python pkg
# 拷贝git仓库的内容到docker内部

# RUN apt-get update && apt-get install -y cpulimit
# RUN apt-get update && apt-get install -y cgroup-tools
# RUN pip install requests -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
# RUN pip install sentence_transformers -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
# 启动服务
# CMD bash /bert_nlu_classify/start_service.sh

WORKDIR /opt/aispeech/
STOPSIGNAL SIGQUIT
EXPOSE 25988
CMD /opt/aispeech/run.sh
