# Multi-Camera Person Tracking and Re-Identification

This project implements a multi-camera person tracking and re-identification system. The system is designed to detect, track, and re-identify individuals across multiple video streams captured from different camera angles.

## Project Structure

```
multi-camera-person-tracking
├── src
│   ├── detection          # Contains YOLO model for object detection
│   ├── feature_extraction # Contains feature extraction logic
│   ├── tracking           # Contains DeepSORT algorithm for tracking
│   ├── reid               # Contains re-identification logic
│   ├── utils              # Contains utility functions for video processing and visualization
│   └── main.py            # Entry point for the application
├── data
│   ├── videos             # Sample video inputs from different cameras
│   └── models             # Pre-trained model weights
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── .gitignore             # Files to ignore in version control
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multi-camera-person-tracking.git
   cd multi-camera-person-tracking
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your video files in the `data/videos` directory.
2. Ensure the pre-trained model weights are in the `data/models` directory.
3. Run the application:
   ```
   python src/main.py
   ```

## Components

- **Object Detection**: Utilizes the YOLO model to detect people in video frames.
- **Feature Extraction**: Extracts features from detected individuals using a deep learning-based feature extractor.
- **Tracking**: Implements the DeepSORT algorithm to track detected individuals across frames.
- **Re-Identification**: Matches and associates detected individuals across multiple cameras using a feature-based similarity metric.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.