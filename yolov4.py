#yolov4.py
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.utils import bbox_iou
import core.common as common
import core.backbone as backbone
from core.config import cfg
from tensorflow.keras import regularizers

# Constants
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
XYSCALE = cfg.YOLO.XYSCALE
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

def YOLO(input_layer, NUM_CLASS, model='yolov4', is_tiny=False, csp_variant='standard'):
    """
    根据指定的模型类型 (model) 和是否Tiny (is_tiny)，调用不同的YOLO结构。
    同时可以选择在 csp_variant 参数中切换 'lite' / 'deeper' / 'standard'（原始）等变体。
    """
    if is_tiny:
        if model == 'yolov4':
            return YOLOv4_tiny(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if model == 'yolov4':
            if csp_variant == 'lite':
                return YOLOv4_lite(input_layer, NUM_CLASS)
            elif csp_variant == 'deeper':
                return YOLOv4_deeper(input_layer, NUM_CLASS)
            else:
                # 默认：原版CSPDarknet53
                return YOLOv4(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3(input_layer, NUM_CLASS)


# =============== YOLOv4 原版：使用 CSPDarknet53 ===============
def YOLOv4(input_layer, NUM_CLASS):
    """原版YOLOv4，主干使用 backbone.CSPDarknet53。"""
    route_1, route_2, conv = backbone.CSPDarknet53(input_layer)

    # ========== SPP Module ==========
    conv = common.convolutional(conv, (1, 1, 512, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 512, 1024), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 1024, 512), kernel_regularizer=regularizers.l2(0.0005))

    maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=13, strides=1, padding='SAME')(conv)
    maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=9, strides=1, padding='SAME')(conv)
    maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding='SAME')(conv)

    conv = tf.concat([maxpool_1, maxpool_2, maxpool_3, conv], axis=-1)
    conv = common.convolutional(conv, (1, 1, 2048, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 512, 1024), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 1024, 512), kernel_regularizer=regularizers.l2(0.0005))

    # ========== PANet 分支1 (大感受野输出) ==========
    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024), kernel_regularizer=regularizers.l2(0.0005))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    # ========== PANet 分支2 (中感受野输出) ==========
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)
    conv = common.convolutional(conv, (1, 1, 768, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    # ========== PANet 分支3 (小感受野输出) ==========
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)
    conv = common.convolutional(conv, (1, 1, 384, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


# =============== YOLOv4 Lite：使用 CSPDarknetLite ===============
def YOLOv4_lite(input_layer, NUM_CLASS):
    """
    YOLOv4 的“轻量化”示例版本，主干换成 backbone.CSPDarknetLite。
    其余结构（SPP、PANet等）与原版YOLOv4相同。
    """
    # 使用更轻量的CSPDarknetLite
    route_1, route_2, conv = backbone.CSPDarknetLite(input_layer)

    # SPP
    conv = common.convolutional(conv, (1, 1, 256, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))

    maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=13, strides=1, padding='SAME')(conv)
    maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=9, strides=1, padding='SAME')(conv)
    maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding='SAME')(conv)

    conv = tf.concat([maxpool_1, maxpool_2, maxpool_3, conv], axis=-1)
    conv = common.convolutional(conv, (1, 1, 1024, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))

    # PANet 分支1
    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    # PANet 分支2
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)
    conv = common.convolutional(conv, (1, 1, 128 + 256, 128), kernel_regularizer=regularizers.l2(0.0005))  # 384->128
    conv = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    # PANet 分支3
    conv = common.convolutional(conv, (1, 1, 128, 64), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)
    conv = common.convolutional(conv, (1, 1, 64 + 128, 64), kernel_regularizer=regularizers.l2(0.0005))  # 192->64
    conv = common.convolutional(conv, (3, 3, 64, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 128, 64), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 64, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 128, 64), kernel_regularizer=regularizers.l2(0.0005))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 64, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 128, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


# =============== YOLOv4 Deeper：使用 CSPDarknetDeeper ===============
def YOLOv4_deeper(input_layer, NUM_CLASS):
    """
    YOLOv4 的“更深”示例版本，主干换成 backbone.CSPDarknetDeeper。
    其余结构（SPP、PANet等）与原版YOLOv4相同。
    """
    # 使用更深的CSPDarknetDeeper
    route_1, route_2, conv = backbone.CSPDarknetDeeper(input_layer)

    # SPP
    conv = common.convolutional(conv, (1, 1, 512, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 512, 1024), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 1024, 512), kernel_regularizer=regularizers.l2(0.0005))

    maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=13, strides=1, padding='SAME')(conv)
    maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=9, strides=1, padding='SAME')(conv)
    maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding='SAME')(conv)

    conv = tf.concat([maxpool_1, maxpool_2, maxpool_3, conv], axis=-1)
    conv = common.convolutional(conv, (1, 1, 2048, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 512, 1024), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 1024, 512), kernel_regularizer=regularizers.l2(0.0005))

    # PANet 分支1
    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024), kernel_regularizer=regularizers.l2(0.0005))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    # PANet 分支2
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)
    conv = common.convolutional(conv, (1, 1, 256 + 512, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 512, 256), kernel_regularizer=regularizers.l2(0.0005))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512), kernel_regularizer=regularizers.l2(0.0005))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    # PANet 分支3
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)
    conv = common.convolutional(conv, (1, 1, 128 + 256, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv = common.convolutional(conv, (1, 1, 256, 128), kernel_regularizer=regularizers.l2(0.0005))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256), kernel_regularizer=regularizers.l2(0.0005))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)),
                                      activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def filter_bboxes(predictions, score_threshold=0.3, input_shape=(448, 448)):
    """
    Filter bounding boxes based on a score threshold.

    Args:
        predictions: Model predictions.
        score_threshold: Minimum score to keep a detection.
        input_shape: Shape of the input image.

    Returns:
        Filtered bounding boxes, scores, and classes.
    """
    bboxes, scores, classes = [], [], []
    for prediction in predictions:
        for bbox, score, class_id in zip(prediction[..., :4], prediction[..., 4], prediction[..., 5:]):
            if score > score_threshold:
                # Adjust coordinates to the input shape
                x_min, y_min, x_max, y_max = bbox
                x_min = max(0, int(x_min * input_shape[1]))
                y_min = max(0, int(y_min * input_shape[0]))
                x_max = min(input_shape[1], int(x_max * input_shape[1]))
                y_max = min(input_shape[0], int(y_max * input_shape[0]))

                bboxes.append([x_min, y_min, x_max, y_max])
                scores.append(score)
                classes.append(np.argmax(class_id))
    return bboxes, scores, classes

def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE):
    """
    Decode training predictions.
    """
    conv_output = tf.reshape(conv_output, 
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) / tf.cast(output_size, tf.float32)
    
    # 限制 conv_raw_dwdh 的值，防止 tf.exp 溢出
    conv_raw_dwdh = tf.clip_by_value(conv_raw_dwdh, -10.0, 10.0)
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) / tf.cast(output_size, tf.float32)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    return pred_xywh  # 仅返回边界框坐标

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    """
    TensorFlow 框架下的解码函数。

    Args:
        conv_output (tf.Tensor): 卷积输出
        output_size (int): 输出特征图尺寸
        NUM_CLASS (int): 类别数量
        STRIDES (list or np.ndarray): 步幅列表
        ANCHORS (list): 锚点列表
        i (int): 当前尺度索引
        XYSCALE (list): XY 缩放列表

    Returns:
        tuple: 解码后的坐标和概率
    """
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) / tf.cast(output_size, tf.float32)
    
    # 限制 conv_raw_dwdh 的值，防止 tf.exp 溢出
    conv_raw_dwdh = tf.clip_by_value(conv_raw_dwdh, -10.0, 10.0)
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) / tf.cast(output_size, tf.float32)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob


# ==== Focal Loss Implementation ====
def focal_loss_fixed(gamma=3.0, alpha=0.75):
    def focal_loss_inner(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1, keepdims=True)  # 保持维度一致
    return focal_loss_inner
    
# ==== Modified Compute Loss to Incorporate Focal Loss ====
def compute_loss(pred, conv, label, bboxes, bboxes_mask,
                 STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i,
                 bbox_giou_fn):
    """
    计算损失函数，包括 GIoU 损失、置信度 (Focal Loss) 和类别概率损失。
    """
    epsilon = 1e-7
    # ============ 1) 获取对象掩码与预测框 ============
    object_mask = label[..., 4:5]   # 正例=1, 负例=0
    pred_xywh = pred               # [batch,grid,grid,3,4]
    label_xywh = label[..., 0:4]   # [batch,grid,grid,3,4]

    # ============ 2) 计算 GIoU 损失 ============
    giou = tf.expand_dims(bbox_giou_fn(pred_xywh, label_xywh), axis=-1)
    giou = tf.where(tf.math.is_finite(giou), giou, tf.zeros_like(giou))

    bbox_wh = label_xywh[..., 2:4]
    bbox_area = bbox_wh[..., 0] * bbox_wh[..., 1]
    input_size = tf.cast(STRIDES[i] * tf.shape(conv)[1], tf.float32)
    bbox_area_norm = bbox_area / (input_size**2 + epsilon)
    bbox_loss_scale = 2.0 - bbox_area_norm
    bbox_loss_scale = tf.expand_dims(bbox_loss_scale, axis=-1)

    giou_loss = object_mask * bbox_loss_scale * (1.0 - giou)
    # 做reduce
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    # 确保有限
    giou_loss = tf.where(tf.math.is_finite(giou_loss), giou_loss, tf.zeros_like(giou_loss))

    # ============ 3) 置信度损失: Focal Loss + 正例加权 ============
    conf_logits = conv[..., 4:5]    # [batch,grid,grid,3,1]
    conf_labels = object_mask       # 1(正例)/0(负例)
    
    focal_fn = focal_loss_fixed(gamma=3.0, alpha=0.25)   # 你定义的focal
    focal_loss_map = focal_fn(conf_labels, tf.sigmoid(conf_logits))  
    # shape: [batch,grid,grid,3,1]

    # 给正例再额外加权
    pos_weight = 4.0
    weighted_focal = (
        pos_weight * conf_labels * focal_loss_map
        + (1.0 - conf_labels) * focal_loss_map
    )
    conf_loss = tf.reduce_mean(tf.reduce_sum(weighted_focal, axis=[1,2,3,4]))

    # 再检查 finite
    conf_loss = tf.where(tf.math.is_finite(conf_loss), conf_loss, tf.zeros_like(conf_loss))

    # ============ 4) 类别概率损失 ============
    prob_logits = conv[..., 5:]      # [batch,grid,grid,3,num_class]
    label_prob  = label[..., 5:]     # 同维度
    prob_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label_prob, logits=prob_logits
    )
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    prob_loss = tf.where(tf.math.is_finite(prob_loss), prob_loss, tf.zeros_like(prob_loss))

    return giou_loss, conf_loss, prob_loss




