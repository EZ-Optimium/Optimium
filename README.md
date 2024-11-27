<div style="text-align: center;">
  <img src="https://github.com/EZ-Optimium/Optimium/blob/main/optimium-brand-signiture-black.png?raw=true" alt="optimiumLogo" width="50%"/>
</div>

# 

Optimium is an AI inference engine that helps users maximize their AI model inference performance. It seamlessly optimizes and deploys models on target devices without requiring any engineering effort. 

Optimium currently supports inference acceleration of computer vision models on x86, x64, Arm64 CPUs - within this scope, it outperforms any other inference options. We plan to expand our coverage to Transformer-based models and GPUs soon, so stay tuned! 📻

# Goal

The purpose of this repository is to share models that are already optimzed using Optimium. If you wish to try it out for your own models and target hardware, sign up for Optimium beta [here](https://wft8y29gq1z.typeform.com/apply4optimium).

We plan to share more & more Optimium-optimized models and your interest will be our priority. Feel free to let us know which models & target hardware you'd like us to optimized via [Discussion](https://github.com/EZ-Optimium/Optimium/discussions)! 

# Performance

Below are performance benchmarks of the models we share in this repository.

### Mediapipe models optimized on Rasperry Pi 5

| Model        | XNNPACK(μs) | Optimium(μs) | No. of Threads | Improvement |
| ---------- | ------------- |------------- |-------------|-------------| 
| Face Detection Short      | 2,967    | 2,146    | 1    | 38.3%    | 
| Face Detection Full      | 14,332    | 10,699    | 1    | 34.0%    |
| Iris Landmark      | 5,054    | 3,605    | 1    | 40.2%    |
| Face Landmark      | 6,602    | 2,846    | 1    | 132.0%    |
| Hand Landmark Lite | 15,729    | 11,635    | 1    | 35.2%    |
| Hand Landmark Full | 27,750    | 24,076    | 1    | 15.3%    |
| Palm Detection Lite | 37,031   | 20,269    | 1    | 82.7%    |
| Palm Detection Full | 43,754   | 23,220    | 1    | 88.4%    |
| Pose Detection      | 31,020   | 25,422    | 1    | 22.0%    |
| Pose Landmark Lite  | 35,047   | 24,720    | 1   | 41.8%    |
| Pose Landmark Full  | 51,429   | 37,411    | 1    | 37.5%    |
| MObileNetV3  | 16,905   | 13,254    | 1    | 27.5%    |





