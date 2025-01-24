import tensorflow as tf
import numpy as np
import cv2
import os

from core.config import cfg
import core.utils as utils
from PIL import Image, ImageDraw, ImageFont

class Dataset:
    def __init__(self, annotations, batch_size, input_size, cfg, is_training=True, drop_remainder=True, dataset_type='yolo'):
        """
        初始化 Dataset 类。

        Args:
            annotations (list): 注释列表，每个元素为 (image_path, bboxes)。
                                其中 bboxes 的shape=[N,5]，格式：[class_id, x_min, y_min, x_max, y_max]
            batch_size (int): 批大小。
            input_size (list or tuple): 输入图像大小，格式为 [height, width, channels]。
            cfg (object): 配置对象（内含 YOLO.STRIDES, YOLO.ANCHORS, YOLO.NUM_CLASS 等）。
            is_training (bool): 是否为训练集，用于数据增强。
            drop_remainder (bool): 是否丢弃最后一个不完整的批次。
            dataset_type (str): 数据集类型，默认为 "yolo"。
        """
        self.annotations = annotations
        self.batch_size = batch_size
        self.input_size = input_size  # [height, width, channels]
        self.cfg = cfg
        self.is_training = is_training
        self.drop_remainder = drop_remainder
        self.dataset_type = dataset_type

        # 从 cfg 加载部分必要的 YOLO 参数（以便后续生成 targets）
        self.STRIDES, self.ANCHORS, self.NUM_CLASS, self.XYSCALE = utils.load_config(cfg)
        # 如果你想使用多尺度 (small, medium, large)，可在 parse_annotation 中遍历 self.STRIDES 等，
        # 并生成多个 targets。本示例仅演示单尺度。

    def load_image_and_bboxes(self, annotation):
        """
        加载图像和边界框，并进行必要的预处理和数据增强。

        Args:
            annotation (tuple): (image_path, bboxes)

        Returns:
            tuple: (image, bboxes)
        """
        image_path, bboxes = annotation
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unable to read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 数据增强
        if self.is_training:
            # 1) 随机水平翻转
            image, bboxes = utils.random_horizontal_flip(image, bboxes)
            # 2) 随机平移
            image, bboxes = utils.random_translate(image, bboxes)
            # 3) 随机裁剪
            image, bboxes = utils.random_crop(image, bboxes)
            # 4) 数据增强结束后，再将图像resize到模型的输入大小
            image, bboxes = utils.image_resize(image, self.input_size[0], bboxes)
            # 也可以尝试先做各种增强，再做一次最终的 image_preprocess()
            # 具体可以根据需求来调整逻辑。
        else:
            # 验证或测试时，只做标准的预处理（resize + 归一化）
            image, bboxes = utils.image_preprocess(image, self.input_size[0], bboxes)

        return image, bboxes

    def parse_annotation(self, annotation):
        """
        解析单个注释，并返回预处理后的图像和目标。

        Args:
            annotation (tuple): (image_path, bboxes)

        Returns:
            tuple: (image, yolo_target, bboxes)
                - image: 处理后的图像 (H,W,3)
                - yolo_target: YOLO 格式的训练标签 (grid_size, grid_size, num_anchors, 5+num_class)
                - bboxes: 原始坐标或增强后的坐标 (N,5)，仅便于调试或可视化
        """
        # 1) 加载图像并做增强 / 预处理
        image, bboxes = self.load_image_and_bboxes(annotation)

        # 2) 若是 YOLO 类型的数据，则生成 YOLO 训练需要的 target
        if self.dataset_type == 'yolo':
            # 为简单演示，这里只使用 self.STRIDES[0], self.ANCHORS[0] 生成单尺度 target
            # 若需要多尺度，请自行根据 len(self.STRIDES) 做循环
            stride = self.STRIDES[0]
            anchors = self.ANCHORS[0]
            num_class = self.NUM_CLASS

            # 调用 utils.generate_targets 生成此尺度下的标签
            # generate_targets(bboxes, input_size, stride, anchors, num_class)
            yolo_target = utils.generate_targets(
                bboxes, self.input_size[0], stride, anchors, num_class
            )
            # 转成 float32
            yolo_target = yolo_target.astype(np.float32)

            return image, yolo_target, bboxes
        else:
            # 如果还有其他数据集类型，可在此根据需要返回对应的 target
            # 暂时仅返回 (image, bboxes)
            return image, bboxes

    def get_dataset(self):
        """
        创建 TensorFlow 数据集，并将其返回。

        Returns:
            tf.data.Dataset: 处理后的数据集，元素格式根据 parse_annotation 的输出而定。
        """
        # 1) 从注释列表创建 Dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.annotations)

        # 2) 如果是训练集，可以加 shuffle
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=len(self.annotations))

        # 3) 使用 py_function 调用 parse_annotation 进行图像读取和预处理
        if self.dataset_type == 'yolo':
            # 对应 parse_annotation 的返回: (image, yolo_target, bboxes)
            # => types: (tf.float32, tf.float32, tf.float32)
            dataset = dataset.map(
                lambda x: tf.py_function(
                    self.parse_annotation,
                    [x],
                    [tf.float32, tf.float32, tf.float32],
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # 设置图像和标签的静态形状（可根据实际需要再改）
            # 注意：image shape=[input_size[0], input_size[1], 3]
            #       yolo_target shape=[grid, grid, num_anchors, 5+num_class]
            #       bboxes shape=[None, 5]，动态
            # 这里只能大致 set_shape 或使用 reshape；
            # bboxes 通常可以是可变长度，因此不设置固定 shape。
            stride = self.STRIDES[0]
            num_anchors = len(self.ANCHORS[0])
            num_class = self.NUM_CLASS

            grid_size = self.input_size[0] // stride
            dataset = dataset.map(
                lambda image, target, bboxes: (
                    tf.reshape(image, self.input_size),  # [H, W, 3]
                    tf.reshape(
                        target,
                        (grid_size, grid_size, num_anchors, 5 + num_class)
                    ),
                    bboxes
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # 若不是 yolo，可根据 parse_annotation 的返回值进行 map
            # 这里 parse_annotation 返回 (image, bboxes)
            dataset = dataset.map(
                lambda x: tf.py_function(
                    self.parse_annotation, [x], [tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.map(
                lambda image, bboxes: (
                    tf.reshape(image, self.input_size),
                    bboxes
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # 4) 批处理
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)

        # 5) 预取
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

