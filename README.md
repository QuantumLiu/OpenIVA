English | [中文](README_zh.md)
# OpenIVA
OpenIVA is an end-to-end intelligent video analytics development toolkit based on different inference backends, designed to help individual users and start-ups quickly launch their own video AI services.  
OpenIVA implements varied mainstream facial recognition, object detection, segmentation and landmark detection algorithms. And it provides an efficient and lightweight service deployment framework with a modular design. Users only need to replace the algorithm model used for their own tasks.
# Features
1. Common mainstream algorithms
- Provides latest fast accurate pre-trained models for facial recognition, object detection, segmentation and landmark detection tasks
2. Multi inference backends
- Supports TensorlayerX/ TensorRT/ onnxruntime
3. High performance  
- Achieves high performance on CPU/GPU/Ascend platforms, achieve inference speed above 3000it/s
4. Asynchronous & multithreading
- Use multithreading and queue to achieve high device utilization for inference and pre/post-processing
5. Lightweight service
- Use Flask for lightweight intelligent application services
6. Modular design 
- You can quickly start your intelligent analysis service, only need to replace the AI models
7. GUI visualization tools  
- Start analysis tasks only by clicking buttons, and show visualized results in GUI windows, suitable for multiple tasks

![alt Sample Face landmark](datas/imgs_results/vis_landmark.jpg)
![alt Sample Face recognition](datas/imgs_results/vis_recog.jpg)

# Performance benchmark
## Testing environments 
- i5-10400 6c12t
- RTX3060  
- Ubuntu18.04
- CUDA11.1
- TensorRT-7.2.3.4
- onnxruntime with EPs:
  - CPU(Default)
  - CUDA(Compiled)
  - OpenVINO(Compiled)
  - TensorRT(Compiled)

## Performance
### Facial recognition
Run  
`python test_landmark.py`  
`batchsize=8`, 67 faces in the image
- Face detection(faces per sec)  
  Model `face_detector_640_dy_sim`
  - CPU :  2075
  - OpenVINO : 5374
  - CUDA : 6972
  - TensorRT(FP32) : 7948
  - TensorRT(FP16) : 8527

- Face landmark (faces per sec)  
  Model `landmarks_68_pfld_dy_sim`
  - CPU : 69
  - OpenVINO : 819
  - CUDA : 2061
  - TensorRT(FP32) : 2639
  - TensorRT(FP16) : 3131 

- Face embedding (faces per sec)  
  Model `arc_mbv2_ccrop_sim`
  - CPU : 212
  - OpenVINO : 865
  - CUDA : 1679
  - TensorRT(FP32) : 2132
  - TensorRT(FP16) : 2744 

# Progress  
- [ ] Multi inference backends
    - [x] onnxruntime
        - [x] CPU
        - [x] CUDA
        - [x] TensorRT
        - [x] OpenVINO
    - [ ] TensorlayerX
    - [ ] TensorRT
- [ ] Asynchronous & multithreading
    - [x] prototype

- [ ] Lightweight service
    - [x] prototype

- [ ] GUI visualization tools

- [ ] Common algorithms
    - [x] Facial recognition
      - [x] Face detection

      - [x] Face landmark

      - [x] Face embedding
    
    - [ ] Object detection
      - [ ] YOLOX
    - [ ] Semantic/Instance segmentation

    - [ ] Scene classification
        - [x] prototype
