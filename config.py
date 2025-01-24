from easydict import EasyDict as edict
import os

# Initialize configuration dictionary
__C = edict()
cfg = __C  # Consumers can import `cfg` as: from core.config import cfg

# YOLO Options
__C.YOLO = edict()

# Path to class names file
__C.YOLO.CLASSES = "./data/classes.txt"
__C.YOLO.NUM_CLASS = 2  # 设置类别数为2

# YOLOv4 anchors for different object scales
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHORS = [
    [(12, 16), (19, 36), (40, 28)],    # Scale 0 (stride 8)
    [(36, 75), (76, 55), (72, 146)],   # Scale 1 (stride 16)
    [(142, 110), (192, 243), (459, 401)]  # Scale 2 (stride 32)
]
# Anchor mask for YOLOv4 layers (three layers)
__C.YOLO.ANCHOR_MASK = [[6,7,8], [3,4,5], [0,1,2]]

# YOLOv3 anchors (use if running YOLOv3 model)
__C.YOLO.ANCHORS_V3 = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]

# YOLOv4-Tiny anchors (use if running YOLOv4-Tiny model)
__C.YOLO.ANCHORS_TINY = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]

# Strides for YOLOv4-Tiny
__C.YOLO.STRIDES_TINY = [16, 32]

# XYSCALE values for YOLOv4 (for bounding box adjustments)
__C.YOLO.XYSCALE = [1.05, 1.05, 1.05]

# XYSCALE values for YOLOv4-Tiny
__C.YOLO.XYSCALE_TINY = [1.05, 1.05]

# Number of anchors per scale
__C.YOLO.ANCHOR_PER_SCALE = 3

# IOU loss threshold for object detection
__C.YOLO.IOU_LOSS_THRESH = 0.5

# Model type: 'yolov4' or 'yolov3'
__C.YOLO.MODEL = 'yolov4'

# Tiny model flag (set to True for YOLOv4-Tiny, False for YOLOv4)
__C.YOLO.TINY = False

# Training Options
__C.TRAIN = edict()

# Path to the training annotations
__C.TRAIN.ANNOT_PATH = "./data/train3.txt"

__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.INPUT_SIZE = 448
__C.TRAIN.DATA_AUG = True

# Learning rate settings
__C.TRAIN.LR_INIT = 1e-4
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 5        # 修改为5
__C.TRAIN.FIRST_STAGE_EPOCHS = 20  # 修改为20
__C.TRAIN.SECOND_STAGE_EPOCHS = 80 # 修改为80（确保总共100 epochs）

# Path to pre-trained weights (if any)
__C.TRAIN.WEIGHTS_PATH = None

# Validation Options
__C.VALID = edict()

__C.VALID.ANNOT_PATH = "./data/val3.txt"

__C.VALID.BATCH_SIZE = 32
__C.VALID.INPUT_SIZE = 448
__C.VALID.DATA_AUG = False

# Testing Options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "./data/test3.txt"

__C.TEST.BATCH_SIZE = 32
__C.TEST.INPUT_SIZE = 448
__C.TEST.DATA_AUG = False

# Detection output and threshold settings
__C.TEST.DETECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD = 0.5
__C.TEST.IOU_THRESHOLD = 0.3

