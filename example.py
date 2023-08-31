from odin.model import Odin

model = Odin(
    source_weights_path="https://drive.google.com/file/d/10U_gu5Wm3xc8tbGNJ-HDNphttUzTIQX6/view?usp=drive_link",
    source_video_path="input_video.mp4",
    target_video_path="output_video.mp4",
    confidence_threshold=0.5,
    iou_threshold=0.6
)

model.run()