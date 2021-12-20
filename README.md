English | [中文](README_zh.md)
# OpenIVA
OpenIVA is an end-to-end intelligent video analytics development toolkit based on different inference backends, designed to help individual users and start-ups quickly launch their own video AI services.  
OpenIVA implements varied mainstream facial recognition, object detection, segmentation and landmark detection algorithms. And it provides an efficient and lightweight service deployment framework with a modular design. Users only need to replace the algorithm model used for their own tasks.
# Features
1. Common mainstream algorithms
- Provides latest fast accurate pre-trained models for facial recognition, object detection, segmentation and landmark detection tasks
2. Multi inference backends
- Supports TensorlayerX/ TensorRT/ onnxruntime
- Achieves high performance on CPU/GPU/Ascend platforms
3. Asynchronous & multithreading
- Use multithreading and queue to achieve high device utilization for inference and pre/post-processing
4. Lightweight service
- Use Flask for lightweight intelligent application services
5. Modular design 
- You can quickly start your intelligent analysis service, only need to replace the AI models
6. GUI visualization tools  
- Start analysis tasks only by clicking buttons, and show visualized results in GUI windows, suitable for multiple tasks
# Progress  
- [] Multi inference backends
    - [x] onnxruntime
        - [x] CPU
        - [x] CUDA
        - [x] TensorRT
        - [x] OpenVINO
    - [] TensorlayerX
    - [] TensorRT
- [] Asynchronous & multithreading
    - [x] prototype

- [] Lightweight service
    - [x] prototype

- [] GUI visualization tools

- [] Common algorithms
    - [x] Face detection

    - [x] Face landmark

    - [] Face embedding
        - [x] prototype
    
    - [] Object detection
    - [] Semantic/Instance segmentation

    - [] Scene classification
        - [x] prototype
