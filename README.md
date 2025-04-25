# ROS 2 MicroSegNet Inference

A ROS 2 packages for performing real-time prostate US segmentation inference using [MicroSegNet](https://github.com/Tony607/MicroSegNet), optimized for robotics applications.

## 📦 Features

- Real-time instance segmentation with combined classification model and MicroSegNet segmentation model
- ROS 2 integration with image input/output topics
- Integration with KUKA LBR robotic arm - the combined information of US probe angle and US image
- Storing the segmentation results for 3D object visualization


## 🛠️ Requirements

- ROS 2 (tested on Humble)
- Python 3.8+
- OpenCV
- NumPy
- Pytorch
- TransUNet model file




