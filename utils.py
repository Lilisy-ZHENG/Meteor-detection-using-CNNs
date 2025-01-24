import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans  # 用于锚框生成
import struct

def read_class_names(class_file_name):
    """读取类别名称文件。"""
    with open(class_file_name, 'r') as f:
        class_names = f.read().splitlines()
    class_dict = {i: name for i, name in enumerate(class_names)}
    return class_dict


def binary_focal(y_true, y_pred, gamma=2, alpha=0.25):
    """计算二元分类的焦点损失（Focal Loss）。"""
    y_pred_prob = tf.sigmoid(y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t = y_true * y_pred_prob + (1 - y_true) * (1 - y_pred_prob)
    focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
    bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = focal_weight * bce_loss
    return loss


def get_anchors(anchors):
    """解析配置文件中的锚点信息（仍返回 NumPy 数组列表）。"""
    return [np.array(scale, dtype=np.float32) for scale in anchors]


def load_config(cfg):
    """加载 YOLOv4 的配置。"""
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = get_anchors(cfg.YOLO.ANCHORS)  # 后续可在 train.py 里再转成 tf.constant
    NUM_CLASS = cfg.YOLO.NUM_CLASS
    XYSCALE = cfg.YOLO.XYSCALE if hasattr(cfg.YOLO, 'XYSCALE') else [1.05, 1.05, 1.05]
    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE


def load_freeze_layer(model_type, tiny, csp_variant='standard'):
    """
    根据 model_type、tiny 和 csp_variant，返回需要冻结的层前缀列表。
    """
    if model_type == 'yolov4' and not tiny:
        # 标准版 YOLOv4
        freeze_prefixes_standard = [
            'darknet53_conv1',
            'darknet53_conv2',
            'csp_darknet53_block1',
            'darknet53_conv3',
            'csp_darknet53_block2',
            'darknet53_conv4',
            'csp_darknet53_block3',
            'darknet53_conv5',
            'csp_darknet53_block4',
            'darknet53_conv6',
            'csp_darknet53_block5',
        ]
        # 轻量版 YOLOv4 (CSPDarknetLite)
        freeze_prefixes_lite = [
            'darklite_conv',
            'darklite_block',
        ]
        # 更深版 YOLOv4 (CSPDarknetDeeper)
        freeze_prefixes_deeper = [
            'darkdeep_conv',
            'csp_darkdeep_block'
        ]

        if csp_variant == 'standard':
            return freeze_prefixes_standard
        elif csp_variant == 'lite':
            return freeze_prefixes_lite
        elif csp_variant == 'deeper':
            return freeze_prefixes_deeper
        else:
            return freeze_prefixes_standard + freeze_prefixes_lite + freeze_prefixes_deeper

    elif model_type == 'yolov4' and tiny:
        # YOLOv4 Tiny
        return [
            'darknet53_tiny_conv_1',
            'darknet53_tiny_conv_2',
            'darknet53_tiny_residual_block_1',
            'darknet53_tiny_residual_block_2',
            'darknet53_tiny_residual_block_3',
            'darknet53_tiny_residual_block_4',
            'darknet53_tiny_residual_block_5',
        ]
    else:
        return []


def freeze_all(model, layer_name_prefixes):
    """冻结模型中名称以指定前缀开头的所有层。"""
    if isinstance(model, (tf.keras.Model, tf.keras.layers.Layer)):
        for layer in model.layers:
            if any(layer.name.startswith(prefix) for prefix in layer_name_prefixes):
                layer.trainable = False
                print(f"冻结层 {layer.name}")
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                freeze_all(layer, layer_name_prefixes)


def unfreeze_all(layer):
    """解冻模型的所有子层。"""
    layer.trainable = True
    if hasattr(layer, 'layers'):
        for sub_layer in layer.layers:
            unfreeze_all(sub_layer)


def generate_anchors(annotation_file, input_size, num_clusters=9):
    """通过 K-Means 聚类生成 anchors。"""
    boxes = []
    with open(annotation_file, 'r') as f:
        lines = f.read().splitlines()
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            for bbox_str in parts[1:]:
                if not bbox_str:
                    continue
                bbox = bbox_str.split(',')
                if len(bbox) != 5:
                    print(f"警告：无效的 bbox 格式: '{bbox_str}' 在行: {line_num}")
                    continue
                try:
                    class_id = int(float(bbox[0]))
                    x_center = float(bbox[1])
                    y_center = float(bbox[2])
                    w = float(bbox[3])
                    h = float(bbox[4])
                    if w <= 0 or h <= 0:
                        print(f"警告：边界框宽度或高度为非正值在行 {line_num}: '{bbox_str}'")
                        continue
                    image_path = parts[0]
                    if not os.path.exists(image_path):
                        print(f"警告：图像文件不存在: {image_path} 在行 {line_num}")
                        continue
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"警告：无法读取图像: {image_path} 在行 {line_num}")
                        continue
                    height, width, _ = image.shape
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width - 1, x_max)
                    y_max = min(height - 1, y_max)
                    if x_max <= x_min or y_max <= y_min:
                        print(f"警告：边界框宽度或高度为非正值在行 {line_num}: '{bbox_str}'")
                        continue
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    boxes.append([box_w, box_h])
                except ValueError as e:
                    print(f"警告：解析 bbox '{bbox_str}' 时出错在行: {line_num} -> {e}")
                    continue

    if len(boxes) == 0:
        raise ValueError("未找到有效的边界框用于锚框生成。请检查注释文件。")

    boxes = np.array(boxes)
    print(f"总共收集到 {len(boxes)} 个边界框用于生成锚框。")

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(boxes)
    anchors = kmeans.cluster_centers_
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)[::-1]
    anchors = anchors[sorted_indices]
    anchors = anchors.tolist()
    print(f"K-Means 聚类后的 Anchors: {anchors}")
    return anchors


def load_annotations(file_path, num_classes=2, normalized=True, dataset_type="yolo"):
    """
    加载注释信息，并返回 (image_path, bboxes) 列表。
    bboxes shape: [N, 5], 包含 [class_id, x_min, y_min, x_max, y_max]
    """
    annotations = []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        for line_num, line in enumerate(lines, 1):
            jpg_pos = line.find('.jpg')
            if jpg_pos == -1:
                print(f"警告：无效的注释行 {line_num}，缺少'.jpg': {line}")
                continue
            image_path = line[:jpg_pos + 4]
            bboxes_data = line[jpg_pos + 4:].strip()

            if not bboxes_data:
                annotations.append((image_path, np.empty((0, 5), dtype=np.float32)))
                continue

            bboxes = []
            bbox_strs = bboxes_data.split(' ')
            for bbox_str in bbox_strs:
                if not bbox_str:
                    continue
                bbox = bbox_str.split(',')
                if len(bbox) != 5:
                    print(f"警告：无效的 bbox 格式在行 {line_num}: {bbox_str}")
                    continue
                try:
                    class_id = int(float(bbox[0]))
                    if class_id < 0 or class_id >= num_classes:
                        print(f"警告: 无效的 class_id {class_id} 在行 {line_num}, 跳过此 bbox.")
                        continue
                    x_center = float(bbox[1])
                    y_center = float(bbox[2])
                    w = float(bbox[3])
                    h = float(bbox[4])

                    if normalized:
                        if not os.path.exists(image_path):
                            print(f"警告：图像文件不存在: {image_path} 在行 {line_num}")
                            continue
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"警告：无法读取图像: {image_path} 在行 {line_num}")
                            continue
                        height, width, _ = image.shape
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height

                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2

                    if normalized:
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(width - 1, x_max)
                        y_max = min(height - 1, y_max)

                    if x_max <= x_min or y_max <= y_min:
                        print(f"警告：边界框宽度或高度为非正值在行 {line_num}: '{bbox_str}'")
                        continue
                    bboxes.append([class_id, x_min, y_min, x_max, y_max])
                except ValueError as e:
                    print(f"警告：解析 bbox '{bbox_str}' 时出错在行 {line_num}: {e}")
                    continue
            if bboxes:
                annotations.append((image_path, np.array(bboxes, dtype=np.float32)))
            else:
                annotations.append((image_path, np.empty((0, 5), dtype=np.float32)))
    return annotations


def image_preprocess(image, input_size, bboxes):
    """
    将图像缩放到 input_size x input_size 并归一化到 [0,1]，同步缩放 bboxes 坐标。
    """
    original_h, original_w, _ = image.shape
    image = cv2.resize(image, (input_size, input_size))
    image = image / 255.0
    if bboxes.size > 0:
        scale_x = input_size / original_w
        scale_y = input_size / original_h
        bboxes[:, 1] = bboxes[:, 1] * scale_x
        bboxes[:, 3] = bboxes[:, 3] * scale_x
        bboxes[:, 2] = bboxes[:, 2] * scale_y
        bboxes[:, 4] = bboxes[:, 4] * scale_y
        bboxes[:, 1:5] = np.clip(bboxes[:, 1:5], 0.0, input_size - 1)
    return image.astype(np.float32), bboxes.astype(np.float32)


def image_resize(image, input_size, bboxes):
    """数据增强后再次 resize 到原始 input_size 大小."""
    original_h, original_w, _ = image.shape
    image = cv2.resize(image, (input_size, input_size))
    assert image.shape[:2] == (input_size, input_size), f"image_resize failed: {image.shape[:2]} != {(input_size, input_size)}"
    scale_x = input_size / original_w
    scale_y = input_size / original_h
    if bboxes.size > 0:
        bboxes[:, 1] = bboxes[:, 1] * scale_x
        bboxes[:, 3] = bboxes[:, 3] * scale_x
        bboxes[:, 2] = bboxes[:, 2] * scale_y
        bboxes[:, 4] = bboxes[:, 4] * scale_y
        bboxes[:, 1:5] = np.clip(bboxes[:, 1:5], 0.0, input_size - 1)
    return image.astype(np.float32), bboxes.astype(np.float32)


def draw_bboxes(image, bboxes, save_path, class_names=None, bbox_color='red', thickness=2):
    """在图像上绘制边界框并保存."""
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    draw_obj = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except IOError:
        font = ImageFont.load_default()

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        for t in range(thickness):
            draw_obj.rectangle([x_min - t, y_min - t, x_max + t, y_max + t], outline=bbox_color)

        if class_names is not None and int(class_id) < len(class_names):
            label = f"{class_names[int(class_id)]}"
        else:
            label = f"Class {int(class_id)}"

        text_bbox = draw_obj.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw_obj.rectangle([x_min, y_min - text_height, x_min + text_width, y_min], fill=bbox_color)
        draw_obj.text((x_min, y_min - text_height), label, fill='white', font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image_pil.save(save_path)
    print(f"图像已保存到 {save_path}")


def random_horizontal_flip(image, bboxes, prob=0.5):
    """随机水平翻转图像和边界框。"""
    if np.random.rand() < prob:
        image = np.fliplr(image)
        width = image.shape[1]
        if bboxes.size > 0:
            bboxes_flipped = bboxes.copy()
            bboxes_flipped[:, 1] = width - bboxes[:, 3]
            bboxes_flipped[:, 3] = width - bboxes[:, 1]
            bboxes = bboxes_flipped
    return image, bboxes


def random_crop(image, bboxes, crop_size=(0.8, 1.0)):
    """随机裁剪图像和边界框。"""
    height, width, _ = image.shape
    scale = np.random.uniform(crop_size[0], crop_size[1])
    new_h, new_w = int(height * scale), int(width * scale)
    new_h = min(new_h, height)
    new_w = min(new_w, width)
    top = np.random.randint(0, height - new_h) if height > new_h else 0
    left = np.random.randint(0, width - new_w) if width > new_w else 0
    image_cropped = image[top:top + new_h, left:left + new_w, :]
    if bboxes.size > 0:
        bboxes_crop = bboxes.copy()
        bboxes_crop[:, 1] = bboxes[:, 1] - left
        bboxes_crop[:, 3] = bboxes[:, 3] - left
        bboxes_crop[:, 2] = bboxes[:, 2] - top
        bboxes_crop[:, 4] = bboxes[:, 4] - top
        bboxes_crop[:, 1] = np.clip(bboxes_crop[:, 1], 0.0, new_w)
        bboxes_crop[:, 3] = np.clip(bboxes_crop[:, 3], 0.0, new_w)
        bboxes_crop[:, 2] = np.clip(bboxes_crop[:, 2], 0.0, new_h)
        bboxes_crop[:, 4] = np.clip(bboxes_crop[:, 4], 0.0, new_h)
        bbox_w = bboxes_crop[:, 3] - bboxes_crop[:, 1]
        bbox_h = bboxes_crop[:, 4] - bboxes_crop[:, 2]
        valid_indices = (bbox_w > 1) & (bbox_h > 1)
        bboxes_crop = bboxes_crop[valid_indices]
        bboxes = bboxes_crop
    return image_cropped, bboxes


def random_translate(image, bboxes, max_translate=(0.2, 0.2)):
    """随机平移图像和边界框。"""
    height, width, _ = image.shape
    max_tx = max_translate[0] * width
    max_ty = max_translate[1] * height
    translate_x = np.random.uniform(-max_tx, max_tx)
    translate_y = np.random.uniform(-max_ty, max_ty)
    M = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    image_translated = cv2.warpAffine(image, M, (width, height),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    if bboxes.size > 0:
        bboxes_translated = bboxes.copy()
        bboxes_translated[:, 1] += translate_x
        bboxes_translated[:, 3] += translate_x
        bboxes_translated[:, 2] += translate_y
        bboxes_translated[:, 4] += translate_y
        bboxes_translated[:, 1] = np.clip(bboxes_translated[:, 1], 0, width - 1)
        bboxes_translated[:, 3] = np.clip(bboxes_translated[:, 3], 0, width - 1)
        bboxes_translated[:, 2] = np.clip(bboxes_translated[:, 2], 0, height - 1)
        bboxes_translated[:, 4] = np.clip(bboxes_translated[:, 4], 0, height - 1)
        bbox_w = bboxes_translated[:, 3] - bboxes_translated[:, 1]
        bbox_h = bboxes_translated[:, 4] - bboxes_translated[:, 2]
        valid_indices = (bbox_w > 1) & (bbox_h > 1)
        bboxes_translated = bboxes_translated[valid_indices]
        bboxes = bboxes_translated
    return image_translated, bboxes


def validate_bboxes(valid_bboxes):
    """过滤掉宽/高 < 1 的边界框。"""
    widths = valid_bboxes[..., 3] - valid_bboxes[..., 1]
    heights = valid_bboxes[..., 4] - valid_bboxes[..., 2]
    conditions = (widths > 1) & (heights > 1)
    filtered_bboxes = tf.ragged.boolean_mask(valid_bboxes, conditions)
    return filtered_bboxes


def calculate_iou(box1, box2):
    """
    计算 2D IoU, box = [x_min, y_min, x_max, y_max].
    """
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])
    inter_w = max(inter_xmax - inter_xmin, 0)
    inter_h = max(inter_ymax - inter_ymin, 0)
    inter_area = inter_w * inter_h
    box1_area = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    box2_area = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    else:
        return inter_area / union_area


def generate_targets(bboxes, input_size, stride, anchors, num_class=2):
    """
    将一张图像的 GT 分配到当前 stride 特征图上。
    anchors: 形如 [[w1,h1],[w2,h2], ...] 的列表(可为 np.ndarray 或 tf.constant, 只要可索引即可)。
    """
    grid_size = input_size // stride
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_class), dtype=np.float32)

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        box_w = x_max - x_min
        box_h = y_max - y_min

        grid_x = int(x_center // stride)
        grid_y = int(y_center // stride)
        if grid_x >= grid_size or grid_y >= grid_size:
            continue

        best_iou = 0
        best_anchor = 0
        for anchor_idx, anchor_wh in enumerate(anchors):
            # 若 anchor_wh 是 tf.Tensor, 可用 anchor_wh.numpy() 或 tf.* 方式
            if isinstance(anchor_wh, tf.Tensor):
                anchor_wh = anchor_wh.numpy()
            anchor_w, anchor_h = anchor_wh

            iou = calculate_iou(
                [x_min, y_min, x_max, y_max],
                [x_min, y_min, x_min + anchor_w, y_min + anchor_h]
            )
            if iou > best_iou:
                best_iou = iou
                best_anchor = anchor_idx

        target_x = (x_center % stride) / stride
        target_y = (y_center % stride) / stride
        target_w = box_w / stride
        target_h = box_h / stride

        target[grid_y, grid_x, best_anchor, 0:2] = [target_x, target_y]
        target[grid_y, grid_x, best_anchor, 2:4] = [target_w, target_h]
        target[grid_y, grid_x, best_anchor, 4] = 1.0
        target[grid_y, grid_x, best_anchor, 5 + int(class_id)] = 1.0

    return target


# ============== 以下为关键性改动：使用 tf.* API 而非 np.* 在 decode_* 中 ==============

def decode_predictions(pred_result, input_size, STRIDES, ANCHORS, NUM_CLASS,
                       XYSCALE, conf_threshold=0.3, nms_threshold=0.3):
    pred_boxes = []  # 存放每张图像的检测结果

    for i, pred in enumerate(pred_result):
        pred_tensor = tf.convert_to_tensor(pred)  # 转TF

        stride = tf.constant(STRIDES[i], dtype=tf.float32)
        anchors_tf = tf.convert_to_tensor(ANCHORS[i], dtype=tf.float32)  # [3,2]
        scale = tf.constant(XYSCALE[i], dtype=tf.float32)

        grid_size = input_size // STRIDES[i]
        num_anchors = tf.shape(anchors_tf)[0]  # 应该是3
        # 先 reshape 到 [batch, grid, grid, 3, 5+NUM_CLASS]
        pred_tensor = tf.reshape(
            pred_tensor,
            (-1, grid_size, grid_size, num_anchors, 5 + NUM_CLASS)
        )
        batch_size = tf.shape(pred_tensor)[0]

        for b in range(batch_size):
            # shape [grid_size, grid_size, 3, 5+NUM_CLASS]
            box = pred_tensor[b]

            # --- 解码过程在4D形状下进行 ---
            # 1) box_xy
            box_xy = tf.sigmoid(box[..., 0:2])
            # 先乘 stride
            box_xy = box_xy * stride
            # 再乘 xyscale
            box_xy = box_xy * scale

            # 2) box_wh
            # anchors reshape成 [1,1,3,2] 方便广播
            anchors_reshape = tf.reshape(anchors_tf, [1,1,num_anchors,2])
            box_wh = tf.exp(box[..., 2:4]) * anchors_reshape

            # 3) objectness & class_probs
            objectness = tf.sigmoid(box[..., 4:5])  # shape [grid, grid, 3, 1]
            class_probs = tf.sigmoid(box[..., 5:])  # shape [grid, grid, 3, NUM_CLASS]

            # 4) flatten
            grid_area = grid_size * grid_size * num_anchors
            box_xy = tf.reshape(box_xy, [grid_area, 2])
            box_wh = tf.reshape(box_wh, [grid_area, 2])
            objectness = tf.reshape(objectness, [grid_area])
            class_probs = tf.reshape(class_probs, [grid_area, NUM_CLASS])

            # 计算score & classes
            max_class_prob = tf.reduce_max(class_probs, axis=-1)   # [grid_area]
            scores = objectness * max_class_prob
            classes = tf.cast(tf.argmax(class_probs, axis=-1), tf.float32)

            # 计算边界框
            x_min = box_xy[:, 0] - box_wh[:, 0] / 2
            y_min = box_xy[:, 1] - box_wh[:, 1] / 2
            x_max = box_xy[:, 0] + box_wh[:, 0] / 2
            y_max = box_xy[:, 1] + box_wh[:, 1] / 2

            box_coord = tf.stack([x_min, y_min, x_max, y_max, classes, scores], axis=-1)

            # 过滤置信度
            mask = box_coord[..., 5] > conf_threshold
            filtered_box = tf.boolean_mask(box_coord, mask)

            if tf.shape(filtered_box)[0] == 0:
                pred_boxes.append(np.array([]))
                continue

            # NMS
            selected_indices = tf.image.non_max_suppression(
                filtered_box[..., :4],
                filtered_box[..., 5],
                max_output_size=100,
                iou_threshold=nms_threshold,
                score_threshold=conf_threshold
            )
            selected_boxes = tf.gather(filtered_box, selected_indices)

            pred_boxes.append(selected_boxes.numpy())
    return pred_boxes



def decode_train(conv, grid_size, num_class, strides, anchors, i, xyscale):
    """
    训练时使用的decode逻辑。只输出解码后的 [xy, wh, objectness, class_probs]。
    这里同样移除 np.array(anchors[i])，改用 tf.constant anchors[i]。
    """
    conv = tf.convert_to_tensor(conv)
    batch_size = tf.shape(conv)[0]
    num_anchors = tf.shape(anchors[i])[0]

    conv = tf.reshape(conv, (batch_size, grid_size, grid_size, num_anchors, 5 + num_class))

    # 取 stride, anchors, xyscale
    stride_tf = tf.constant(strides[i], dtype=tf.float32)
    anchors_tf = tf.convert_to_tensor(anchors[i], dtype=tf.float32)  # shape [num_anchors,2]
    xysc = tf.constant(xyscale[i], dtype=tf.float32)

    box_xy = tf.sigmoid(conv[..., 0:2]) * stride_tf
    # 额外对 y 乘 xyscale? 如果你想跟 x 一起处理，也可以都乘 xyscale
    # 这里只是示例逻辑
    box_xy = box_xy * xysc

    box_wh = tf.exp(conv[..., 2:4]) * anchors_tf  # 广播乘法

    objectness = tf.sigmoid(conv[..., 4:5])
    class_probs = tf.sigmoid(conv[..., 5:])

    pred = tf.concat([box_xy, box_wh, objectness, class_probs], axis=-1)
    return pred


def bbox_iou(boxes1, boxes2):
    """
    IoU 计算 (tensor 版本).
    boxes shape: [..., 4], 其中 4=[x_min, y_min, x_max, y_max].
    """
    inter_left_top = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_right_bottom = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = tf.maximum(inter_right_bottom - inter_left_top, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    boxes1_wh = tf.maximum(boxes1[..., 2:] - boxes1[..., :2], 0.0)
    boxes2_wh = tf.maximum(boxes2[..., 2:] - boxes2[..., :2], 0.0)
    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
    boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    epsilon = 1e-9
    union_area = tf.maximum(union_area, epsilon)
    iou = inter_area / union_area
    return iou


def bbox_giou(box1, box2):
    """
    计算 GIoU.
    """
    epsilon = 1e-6
    iou = bbox_iou(box1, box2)
    enclosing_min = tf.minimum(box1[..., :2], box2[..., :2])
    enclosing_max = tf.maximum(box1[..., 2:], box2[..., 2:])
    enclosing_wh = tf.maximum(enclosing_max - enclosing_min, 0.0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]

    boxes1_wh = tf.maximum(box1[..., 2:] - box1[..., :2], 0.0)
    boxes2_wh = tf.maximum(box2[..., 2:] - box2[..., :2], 0.0)
    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
    boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]
    inter_area = iou * (boxes1_area + boxes2_area)
    union_area = boxes1_area + boxes2_area - inter_area

    giou = iou - (enclosing_area - union_area) / (enclosing_area + epsilon)
    giou = tf.where(tf.math.is_finite(giou), giou, tf.zeros_like(giou))
    giou = tf.clip_by_value(giou, -1.0, 1.0)
    return giou


def compute_metrics(TP, FP, FN):
    """计算 Precision, Recall, F1."""
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def compute_map(pred_boxes, gt_boxes, num_class, iou_threshold=0.3):
    """
    计算 mAP, 用于评估.
    pred_boxes: list of [N, 6], [x_min, y_min, x_max, y_max, class_id, score]
    gt_boxes: list of [M, 5], [class_id, x_min, y_min, x_max, y_max]
    """
    average_precisions = []

    for c in range(num_class):
        detections = []
        ground_truths = {}

        for img_id in range(len(gt_boxes)):
            gt = gt_boxes[img_id]
            pred = pred_boxes[img_id]
            if len(gt) > 0 and c in gt[:, 0]:
                ground_truths.setdefault(img_id, [])
                ground_truths[img_id] += [bbox for bbox in gt if int(bbox[0]) == c]

            if pred.size > 0:
                preds = pred[pred[:, 4] == c]
                for p in preds:
                    detections.append({
                        'image_id': img_id,
                        'bbox': p[:4],
                        'score': p[5]
                    })

        if len(detections) == 0:
            average_precisions.append(0)
            continue

        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))
        total_gts = sum([len(boxes) for boxes in ground_truths.values()])

        for d, detection in enumerate(detections):
            image_id = detection['image_id']
            bbox_pred = detection['bbox']
            max_iou = 0
            max_gt_idx = -1
            if image_id in ground_truths:
                gt_bboxes = ground_truths[image_id]
                for t, bbox_gt in enumerate(gt_bboxes):
                    iou = calculate_iou(
                        [bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3]],
                        [bbox_gt[1], bbox_gt[2], bbox_gt[3], bbox_gt[4]]
                    )
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = t
            if max_iou >= iou_threshold:
                # 没被匹配过则TP，否则FP
                if 'used' not in ground_truths[image_id][max_gt_idx]:
                    TP[d] = 1
                    ground_truths[image_id][max_gt_idx] = ground_truths[image_id][max_gt_idx].tolist() + ['used']
                else:
                    FP[d] = 1
            else:
                FP[d] = 1

        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        recalls = TP_cumsum / (total_gts + 1e-9)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-9)

        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))

        # 处理 AP 曲线
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        average_precisions.append(ap)

    mAP = np.mean(average_precisions)
    return mAP


def visualize_multiple_test_images(model, class_names, image_paths, STRIDES, ANCHORS, NUM_CLASS,
                                   XYSCALE, input_size, save_dir, ground_truths):
    """
    批量推理并可视化多张测试图像. 
    """
    os.makedirs(save_dir, exist_ok=True)
    for image_path in image_paths:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 预处理图像
        image_resized, _ = image_preprocess(image, input_size, np.empty((0, 5), dtype=np.float32))
        image_input = np.expand_dims(image_resized, axis=0).astype(np.float32)

        # 推理
        preds = model.predict(image_input)

        # 解码预测框 (不再把 preds 强转 NumPy，而是使用 tf.convert_to_tensor)
        pred_boxes_list = decode_predictions(
            preds, input_size, STRIDES, ANCHORS, NUM_CLASS,
            XYSCALE, conf_threshold=0.3, nms_threshold=0.3
        )

        # 对应 batch_size=1, 取 pred_boxes_list[0]
        if len(pred_boxes_list) > 0:
            pred_boxes = pred_boxes_list[0]
        else:
            pred_boxes = np.array([])

        print(f"Image: {image_path}")
        print(f"Decoded pred_boxes before filtering: {pred_boxes}")

        # 过滤 + 取列
        if pred_boxes.size > 0:
            pred_boxes = np.nan_to_num(pred_boxes, nan=0.0, posinf=0.0, neginf=0.0)
            pred_boxes = pred_boxes[pred_boxes[:, 5] > 0.3]  # 过滤低置信度
            print(f"Decoded pred_boxes after filtering: {pred_boxes}")

            # [class_id, x_min, y_min, x_max, y_max]
            pred_boxes = pred_boxes[:, [4, 0, 1, 2, 3]]
        else:
            pred_boxes = np.empty((0, 5), dtype=np.float32)

        gt_boxes = ground_truths.get(image_path, np.empty((0, 5), dtype=np.float32))
        save_path = os.path.join(save_dir, os.path.basename(image_path))

        print(f"Ground truth boxes for {image_path}: {gt_boxes}")

        # 先绘制 GT(绿色)
        if gt_boxes.size > 0:
            draw_bboxes(image_resized, gt_boxes, save_path, class_names=class_names, bbox_color='green')

        # 再绘制预测框(红色)
        if pred_boxes.size > 0:
            draw_bboxes(image_resized, pred_boxes, save_path, class_names=class_names, bbox_color='red')

    print(f"所有可视化图像已保存到 {save_dir}")

