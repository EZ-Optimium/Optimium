<div style="text-align: center;">
  <img src="https://github.com/EZ-Optimium/Optimium/blob/main/optimium-brand-signiture-black.png?raw=true" alt="optimiumLogo" width="50%"/>
</div>

# 

Optimium is an AI inference engine that helps users maximize their AI model inference performance. It seamlessly optimizes and deploys models on target devices without requiring any engineering effort. 

Optimium currently supports inference acceleration of computer vision models on x86, x64, Arm64 CPUs - within this scope, it outperforms any other inference options. We plan to expand our coverage to Transformer-based models and GPUs soon, so stay tuned! 📻
<br>

## Goal

The purpose of this repository is to share models that are already optimzed using Optimium. If you wish to try Optimium out yourself for your own models and target hardware, sign up for Optimium beta [here](https://wft8y29gq1z.typeform.com/apply4optimium). 

We plan to release more Optimium-optimized models and your interest will be our priority. Feel free to let us know which models & target hardware you'd like us to optimize via [Discussion](https://github.com/EZ-Optimium/Optimium/discussions)! 
<br>

## Performance

Below are performance benchmarks of the models we share in this repository.

### Mediapipe models optimized on Rasperry Pi 5(Cortex-A76) - Thread: 1

| Model        | XNNPACK(μs) | Optimium(μs) | Improvement |
| ---------- | ------------- |------------- |-------------| 
| [Face Detection Short](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/face_detection_short.model) | 2,967    | 2,146    |**38.3%**    | 
| [Face Detection Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/face_detection_full.model)  | 14,332    | 10,699    |**34.0%**    |
| [Iris Landmark](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/iris_landmark.model)        | 5,054    | 3,605    |**40.2%**    |
| [Face Landmark](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/face_landmark.model)        | 6,602    | 2,846    |**132.0%**    |
| [Hand Landmark Lite](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/hand_landmark_lite.model)   | 15,729    | 11,635    |**35.2%**    |
| [Hand Landmark Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/hand_landmark_full.model)   | 27,750    | 24,076    |**15.3%**    |
| [Palm Detection Lite](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/palm_detection_lite.model)  | 37,031   | 20,269    |**82.7%**    |
| [Palm Detection Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/palm_detection_full.model)  | 43,754   | 23,220    |**88.4%**    |
| [Pose Landmark Lite](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/pose_landmark_lite.model)   | 35,047   | 24,720    |**41.8%**    |
| [Pose Landmark Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/pose_landmark_full.model)   | 51,429   | 37,411    |**37.5%**    |
| [MobileNetV3](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_1/mobilenetv3_fp32.model)          | 16,905   | 13,254    |**27.5%**    |

### Mediapipe models optimzed on Raspberry Pi 5(Cortex-A76) - Thread: 2
  
| Model        | XNNPACK(μs) | Optimium(μs) | Improvement |
| ---------- | ------------- |------------- |-------------| 
| [Face Detection Short](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/face_detection_short.model) | 1,738    | 1,306    |**33.1%**    | 
| [Face Detection Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/face_detection_full.model)  | 11,007    | 7,418    |**48.4%**    |
| [Iris Landmark](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/iris_landmark.model)        | 3,842    | 2,160    |**77.9%**    |
| [Face Landmark](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/face_landmark.model)        | 5,928    | 2,118    |**179.9%**    |
| [Hand Landmark Lite](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/hand_landmark_lite.model)   | 9,694    | 9,186    |**5.5%**    |
| [Hand Landmark Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/hand_landmark_full.model)   | 16,994    | 16,904    |**0.5%** |  
| [Palm Detection Lite](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/palm_detection_lite.model)  | 25,319   | 12,332    |**105.3%** | 
| [Palm Detection Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/palm_detection_full.model)  | 31,164   | 14,547    |**114.2%**  |
| [Pose Landmark Lite](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/pose_landmark_lite.model)   | 23,946   | 22,813    |**5.0%**    |
| [Pose Landmark Full](https://github.com/EZ-Optimium/Optimium/raw/main/models/mediapipe_thread_2/pose_landmark_full.model)   | 37,666   | 29,763    |**26.6%**    |


## Docs

You'll be able to find out how to install and use Optimium Runtime for the execution of above models [here](https://optimium.readme.io/docs/optimium-runtime-copy).


## Supported layers & architectures

- [Supported layers](https://optimium.readme.io/docs/optimium-copy#supported-layers)
- [Supported architectures](https://optimium.readme.io/docs/optimium-runtime-copy#supported-environments)

## Learn more

- [Website](https://optimium.enerzai.com)
- [Tech Blog](https://medium.com/@enerzai)
- [Docs](https://optimium.readme.io)
- [Demo Video](https://youtu.be/u7wzFngylis)
- [Beta Sign-up](https://wft8y29gq1z.typeform.com/apply4optimium)


