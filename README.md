[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Odin
Super Fast and super cheap object detection at massive scale in less than 10 lines of code!

# Appreciation
* Lucidrains
* Agorians


# Install
`pip install vodin`

# Usage

Here are three examples demonstrating the usage of the `Odin` class from your provided code:

**Example 1: Basic Usage**

```python
# Import the necessary modules and classes
from odin import Odin

# Initialize the Odin object with paths and thresholds
odin = Odin(
    source_weights_path="yolo.weights",
    source_video_path="input_video.mp4",
    target_video_path="output_video.mp4",
    confidence_threshold=0.3,
    iou_threshold=0.7
)

# Run the object to process the video
odin.run()
```

**Example 2: Custom Parameters**

```python
# Import the necessary modules and classes
from odin import Odin

# Initialize the Odin object with custom parameters
odin = Odin(
    source_weights_path="custom_yolo.weights",
    source_video_path="input_video.mp4",
    target_video_path="output_video.mp4",
    confidence_threshold=0.5,
    iou_threshold=0.6
)

# Run the object to process the video
odin.run()
```

**Example 3: Advanced Usage**

```python
# Import the necessary modules and classes
from odin import Odin

# Initialize the Odin object with paths and thresholds
odin = Odin(
    source_weights_path="yolo.weights",
    source_video_path="input_video.mp4",
    target_video_path="output_video.mp4",
    confidence_threshold=0.3,
    iou_threshold=0.7
)

# Customize further configurations if needed
odin.tracker.set_max_distance(50)
odin.box_annotator.set_box_color((0, 255, 0))
odin.model.set_device("cuda")

# Run the object to process the video
odin.run()
```

# Architecture
* [Odin utilizes YoloV7, weights can be downloaded here](https://drive.google.com/file/d/1yEYFq1jCIpklofMMhuqQKwyTfvj1hLQ1/view)

# License
MIT
