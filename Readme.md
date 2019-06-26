# fd-detect

## 简介
缺氟检测项目

## 摘要
- author: 岳冬
- date: 2019/05/22
- company: gree
- email:

## 源码文件说明
 requirements.txt :程序相关依赖,这些信息在./requirements.txt文件中
> 使用pipreqs工具可以生成这个项目的依赖文件列表，命令为：

```shell
pipreqs --force --encoding utf-8 ./
```

安装pipreqs：
```shell
pip install pipreqs
```

再在另一台主机上安装依赖时，可以
```shell
pip install -r requirements.txt
```

参考资料：https://github.com/bndr/pipreqs
有些包安装依赖gcc编译，确保主机上安装有gcc
还可能依赖 python-dev  libd-dev两个包 也通过apt-get安装

## 环境
### python
python版本3.7

### tensorflow
python开源机器学习库，版本1.13.1

### 操作系统
跨平台，支持windows和Linux

## 准备
### MySQL数据库
表定义：
mac | fluorine

host: 172.28.4.107
port: 3306
账户：gree
密码：gree


## 运行方法
源码根目录下，运行命令
```shell
python main.py
# 或者
./main.py
```
即可。

## 部署
### docker
1. 制作镜像
deploy目录下，执行
```shell
docker build -t fd-detect/python:latest .
```

2. 启动容器 
```shell
docker run -d --name=fd-detect --network=host -v /home/yd/fd-detect:/root/fd-detect -w /root/fd-detect fd-detect/python:latest ./main.py
```

或在deploy目录下执行
```shell
docker-compose up -d
```

运行后，查看日志输出：
```
docker logs -f fd-detect
```