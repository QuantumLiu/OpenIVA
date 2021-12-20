[English](README.md) | 中文
# OpenIVA
OpenIVA 是一个端到端的基于多推理后端的智能视频分析开发套件，旨在帮助个人用户和初创企业快速启动自己的视频AI服务。  
OpenIVA实现了各种主流的面部识别、目标检测、分割和关键点检测算法。并且提供了高效的轻量级服务部署框架，采用模块化设计，用户只需要替换用于自己任务的算法模型。
# 特色
1. 常用主流算法
- 提供最新的主流预训练模型，用于面部识别、目标检测、分割和关键点检测等任务
2. 多推理后端
- 支持 TensorlayerX/ TensorRT/ onnxruntime
- 在 CPU/GPU/Ascend 取得高性能表现
3. 异步 & 多线程
- 在推理和预/后处理过程中使用多线程和队列以达到高设备占用率
4. 轻量级服务
- 使用Flask来搭建轻量级智能应用服务
5. 模块化设计
- 你只需要替换AI模型就可以快速启动自己的智能分析服务
6. 图形界面的可视化工具
- 只需要点击几个按钮就可以启动分析任务， 并且可以在GUI窗口里展示可视化的结果，适合多种任务
# 进度  
- [] 多推理后端
    - [x] onnxruntime
        - [x] CPU
        - [x] CUDA
        - [x] TensorRT
        - [x] OpenVINO
    - [] TensorlayerX
    - [] TensorRT
- [] 异步 & 多线程
    - [x] prototype

- [] 轻量级服务
    - [x] prototype

- [] 图形界面的可视化工具

- [] 常用主流算法
    - [x] Face detection

    - [x] Face landmark

    - [] Face embedding
        - [x] prototype
    
    - [] Object detection
    - [] Semantic/Instance segmentation

    - [] Scene classification
        - [x] prototype
