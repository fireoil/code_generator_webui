# <代码生成系统>项目部署文档

## 1 功能

- 代码生成
- 一般的问答功能

## 2 部署条件

- 模型:CodeFuse-DeepSeek-33B-4bits, 可以在huggingface\modelscope等网站下载,具体下载方法见相关网站文档
- docker镜像:code_generator:v1.1

## 3 部署过程

- 加载docker镜像

- 创建容器
- 将模型放到制定位置
- 修改代码中模型位置
- streamlit run webui.py

## 4 界面结果

![](/run/user/1000/doc/6134522e/webui.png)
