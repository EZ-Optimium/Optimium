# Hand Landmarks Detection with Optimium

*Note that this model is optimized for <u>Raspberry PI 5</u> board.

### Prerequisite
- Raspberry PI 5, with Camera
- OpenCV

<br> 

## Install Optimium Runtime

Prepare `optimium-runtime` tar file from this reposity at `Optimium/install/optimium-runtime/cpp.

For Raspberry PI 5, you need file that ends with `arm64-linux.tar.gz`.

Make a directory `.local` in your workspace. You may use different name:

```
cd ~
mkdir .local
```

Untar Optimium Runtime tar file to the directory:
```
tar xf optimium-runtime-0.3.10.dev+05f8377c48-arm64-linux.tar.gz -C .local
```

Add path to `LD_LIBRARY_PATH`. You may add it to `.bashrc` :

```
export LD_LIBRARY_PATH="/home/optima-remote/.local/lib:$LD_LIBRARY_PATH"
```

<br>

## Download Tensorflow Lite
We integrated TFLite into our application for benchmark, to show how Optimium Runtime outperforms in performance.

You may use your own version if you already have installed by modifying `CMakeLists.txt`.

However, it is recommended to simply run script as below:
```
./download-third-party.sh
```

<br>

## Build Application
Simply run script to build our application.

Note that this may take a while since TFLite is linked during build.
```
./build.sh
```

<br>

## Usage
If application build is successfully finished, run application:

```
./build/rpi-demo
```

Once you run the application, you'll see menu:
```
Optimium Demo App Command:
  - 'l' : Live demo mode
  - 'd' : Diffrentiate mode
  - 'r' : Show previous record
  - 'q' : Quit the app

Select the command:
```

### Live Demo mode
If you type 'l', you can run live demo mode. 

This takes camera input frame and run real-time hand landmarks detection at 30 FPS.

Once program started, you can switch mode between **TFLite** and **Optimium** by pressing '**s**'

You can see latency and FPS in the window.

### Differentiate mode
If you type 'd', you can run differentiate mode.

Once you run this mode, you can press 'r' to record video, then 'r' again to stop recoding.

If you type 'q' to quit window then type 'r', you can see slo-mo video that displays both TFLite and Optimium mode.