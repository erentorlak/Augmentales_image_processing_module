# Augmentales Image Processing Module

A real-time card detection system using YOLOv5 and OpenCV for the [Augmentales project](https://github.com/Augmentales). This module captures video from a webcam, detects custom-trained card objects, and sends detection data via WebSocket to external applications.

## 🎯 Features

- **Real-time Object Detection**: Uses a custom-trained YOLOv5 ONNX model to detect 24 different card types
- **Webcam Integration**: Captures live video feed from camera (default: device 0)
- **Frame Buffering**: Implements consistency checking across multiple frames to reduce false positives
- **WebSocket Communication**: Sends detection results to external servers in real-time
- **GPU Acceleration**: Optional CUDA support for improved performance
- **Visual Feedback**: Displays bounding boxes, class labels, confidence scores, and FPS

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Webcam Feed   │───▶│  YOLOv5 Model    │───▶│  Detection      │
│                 │    │  (ONNX Runtime)  │    │  Processing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  WebSocket      │◀───│  Frame Buffer    │◀────────────┘
│  Client         │    │  & Consistency   │
│                 │    │  Checking        │
└─────────────────┘    └──────────────────┘
```

## 🚀 Prerequisites

### System Requirements
- Linux (Ubuntu 18.04+ recommended)
- C++14 compatible compiler
- CMake 3.0 or higher
- Webcam/Camera device

### Dependencies
- **OpenCV 4.2+**: Computer vision library
- **nlohmann/json**: JSON processing
- **WebSocket++**: WebSocket client implementation
- **Boost**: Required for WebSocket++

## 📦 Installation

### 1. Install System Dependencies

```bash
# Update package manager
sudo apt update

# Install build tools
sudo apt install build-essential cmake

# Install OpenCV
sudo apt install libopencv-dev

# Install additional libraries
sudo apt install libboost-all-dev
sudo apt install libwebsocketpp-dev nlohmann-json3-dev
```

### 2. Build the Project

```bash
# Clone the repository (if not already done)
git clone https://github.com/erentorlak/Augmentales_image_processing_module.git
cd Augmentales_image_processing_module

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the application
make

# The executable will be created as 'my_app'
```

## 🎮 Usage

### Basic Usage

1. **Start the WebSocket Server**: First, ensure your WebSocket server is running on `localhost:8080`
   ```bash
   # Example: Start your server (server.mjs or equivalent)
   node server.mjs
   ```

2. **Run the Image Processing Module**:
   ```bash
   # From the build directory
   ./my_app
   
   # For GPU acceleration (if CUDA is available)
   ./my_app cuda
   ```

### Configuration

#### WebSocket Connection
The application connects to a WebSocket server at `ws://localhost:8080` by default. To change this:
- Modify the `uri` variable in the `communication_thread()` function in `inference.cpp`
- Rebuild the application

#### Detection Parameters
Key parameters can be adjusted in `inference.cpp`:
- `CONFIDENCE_THRESHOLD = 0.4`: Minimum detection confidence
- `SCORE_THRESHOLD = 0.2`: Minimum class score threshold  
- `NMS_THRESHOLD = 0.4`: Non-maximum suppression threshold
- `FRAME_BUFFER_SIZE = 10`: Number of frames to buffer
- `CONSISTENCY_THRESHOLD = 7`: Required consistent detections

#### Camera Selection
To use a different camera device, modify the camera index in `main()`:
```cpp
cv::VideoCapture capture(0);  // Change 0 to your camera index
```

## 📡 API Documentation

### WebSocket Communication

The module sends JSON messages via WebSocket with the following format:

```json
{
  "Array": [
    {
      "id": 10,
      "pos_x": 320,
      "pos_y": 240
    },
    {
      "id": 15,
      "pos_x": 450,
      "pos_y": 180
    }
  ]
}
```

#### Message Fields
- `Array`: List of detected cards that meet consistency requirements
- `id`: Transformed card ID (mapped from detected class to game-specific ID)
- `pos_x`: X-coordinate of detection center
- `pos_y`: Y-coordinate of detection center

### Card Classes

The system detects 24 different card types defined in `data/coco.names`:

1. Alien Warrior
2. Amazoness Spiritualist  
3. giski natalia
4. ancient gear golem ultimate pound
5. hebo lord of the river
6. elemental hero heat
7. hino kagu tsuchi
8. ice knight
9. mystical knight of jackal
10. meteo the matchless
11. shinobird pigeon
12. serene psychic witch
13. aqua armor ninja
14. skull flame
15. the Creator Incarnate
16. king of destruction xexex
17. yaksha
18. asset mountis
19. blood sucker
20. choas hunter
21. (additional classes...)

## 🔧 Troubleshooting

### Common Issues

**Camera Not Found**
```
Error opening video capture
```
- Check if camera is connected and accessible
- Try different camera indices (0, 1, 2, etc.)
- Ensure camera permissions are granted

**WebSocket Connection Failed**
```
Could not create connection because: [error message]
```
- Verify WebSocket server is running on localhost:8080
- Check firewall settings
- Ensure server accepts connections

**Model Loading Failed**
```
Failed to load the ONNX model from path: ../yolov5s_custom_model.onnx
```
- Verify the ONNX model file exists in the project root
- Check file permissions
- Ensure the model file is not corrupted

**Missing Dependencies**
```
Package 'opencv4' was not found
```
- Install missing dependencies using the installation commands above
- Check if pkg-config can find the libraries

### Performance Optimization

- **Enable GPU acceleration**: Run with `./my_app cuda` if NVIDIA GPU and CUDA are available
- **Adjust detection thresholds**: Lower thresholds for better sensitivity, higher for better performance
- **Reduce frame buffer size**: Smaller buffers provide faster response but less stability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Project Structure

```
Augmentales_image_processing_module/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── inference.cpp               # Main application source
├── yolov5s_custom_model.onnx  # Trained YOLOv5 model
├── data/
│   └── coco.names             # Class names file
└── build/                     # Build output directory
    └── my_app                 # Compiled executable
```

## 📄 License

This project is part of the Augmentales ecosystem. Please refer to the main project repository for license information.

## 🔗 Related Projects

- [Main Augmentales Project](https://github.com/Augmentales)

---

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/erentorlak/Augmentales_image_processing_module).
