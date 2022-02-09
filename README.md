# NVIDIA Jetson Nano AI-powered Follow Trolley

## Description
An assistive follow trolley project utilising AI running on the NVIDIA Jetson Nano. This code accompanies an article series on [DesignSpark](https://www.rs-online.com/designspark/building-an-ai-powered-follow-trolley-part-1-getting-started)

Original [FastMOT](https://github.com/GeekAlexis/FastMOT) demonstration by GeekAlexis

## Requirements
- CUDA >= 10
- cuDNN >= 7
- TensorRT >= 7
- OpenCV >= 3.3
- PyCuda
- Numpy >= 1.15
- Scipy >= 1.5
- TensorFlow < 2.0 (for SSD support)
- Numba == 0.48
- cython-bbox
- RPi.GPIO

### Install for Jetson (TX2/Xavier NX/Xavier)
Make sure to have [JetPack 4.4+](https://developer.nvidia.com/embedded/jetpack) installed and run the script:
  ```
  $ scripts/install_jetson.sh
  ```
### Install for Ubuntu 18.04
Make sure to have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. The image requires an NVIDIA Driver version >= 450. Build and run the docker image:
  ```
  $ docker build -t fastmot:latest .
  $ docker run --rm --gpus all -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY fastmot:latest
  ```
### Download models
This includes both pretrained OSNet, SSD, and my custom YOLOv4 ONNX model
  ```
  $ scripts/download_models.sh
  ```
### Build YOLOv4 TensorRT plugin
Modify `compute` [here](https://github.com/GeekAlexis/FastMOT/blob/2296fe414ca6a9515accb02ff88e8aa563ed2a05/fastmot/plugins/Makefile#L21) to match your [GPU compute capability](https://developer.nvidia.com/cuda-gpus#compute) for x86 PC
  ```
  $ cd fastmot/plugins
  $ make
  ```
### Download VOC dataset for INT8 calibration
Only required if you want to use SSD
  ```
  $ scripts/download_data.sh
  ```

## Usage
- MIPI CSI camera:
  ```
  $ python3 app.py --input_uri csi://0 -gm
  ```

<details>
<summary> More options can be configured in cfg/mot.json </summary>

  - Set `resolution` and `frame_rate` that corresponds to the source data or camera configuration (optional). They are required for image sequence, camera sources, and MOT Challenge evaluation. List all configurations for your USB/CSI camera:
    ```
    $ v4l2-ctl -d /dev/video0 --list-formats-ext
    ```
  - To change detector, modify `detector_type`. This can be either `YOLO` or `SSD`
  - To change classes, set `class_ids` under the correct detector. Default class is `1`, which corresponds to person
  - To swap model, modify `model` under a detector. For SSD, you can choose from `SSDInceptionV2`, `SSDMobileNetV1`, or `SSDMobileNetV2`
  - Note that with SSD, the detector splits a frame into tiles and processes them in batches for the best accuracy. Change `tiling_grid` to `[2, 2]`, `[2, 1]`, or `[1, 1]` if a smaller batch size is preferred
  - If more accuracy is desired and processing power is not an issue, reduce `detector_frame_skip`. Similarly, increase `detector_frame_skip` to speed up tracking at the cost of accuracy. You may also want to change `max_age` such that `max_age × detector_frame_skip ≈ 30`

</details>

## License
MIT License
2021 GeekAlexis, 2022 RS Components Ltd