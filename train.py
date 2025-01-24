#train.py
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, YOLOv4, decode_train, compute_loss
from core.config import cfg
import numpy as np
import core.utils as utils
from core.utils import calculate_iou
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2  # 确保已安装 OpenCV

# ==== 强制重新加载模块以确保使用最新版本 ====
import core.dataset
import importlib
importlib.reload(core.dataset)
from core.dataset import Dataset

# 清除任何现有的模型并重置层命名
tf.keras.backend.clear_session()
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

# 如果有 GPU 可用，启用内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已为 GPU 启用内存增长。")
    except RuntimeError as e:
        print(e)

# ==== 配置参数 ====
model_type = 'yolov4'
tiny = False  # 定义 tiny 变量
cfg.YOLO.TINY = tiny  # 将 tiny 信息添加到 cfg.YOLO 配置
cfg.TRAIN.WEIGHTS_PATH = None 

# ==== 超参数 ====
batch_size = 32  # 根据 GPU 内存调整
epochs = 100
dataset_path = './data'
classes_path = './data/classes.txt'
output_path = './checkpoints/yolov4_custom'  # 更新为所需的输出路径
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ==== 学习率和损失权重调整 ====
CONF_LOSS_WEIGHT = 2.0  # 增加置信度损失权重

# 调整配置
cfg.YOLO.CLASSES = classes_path
cfg.TRAIN.ANNOT_PATH = os.path.join(dataset_path, 'train3.txt')  # 训练注释文件路径
cfg.VALID.ANNOT_PATH = os.path.join(dataset_path, 'val3.txt')    # 验证注释文件路径
cfg.TEST.ANNOT_PATH = os.path.join(dataset_path, 'test3.txt')    # 测试注释文件路径
cfg.TRAIN.BATCH_SIZE = batch_size
cfg.TRAIN.INPUT_SIZE = 448  # 确保这是一个整数
cfg.TRAIN.DATA_AUG = True

# 修改的学习率参数
cfg.TRAIN.LR_INIT = 1e-5  # 减小初始学习率
cfg.TRAIN.LR_END = 1e-6
cfg.TRAIN.WARMUP_EPOCHS = 5 # 将 warmup epochs 从 2 增加到 5

cfg.TRAIN.FIRST_STAGE_EPOCHS = 20
cfg.TRAIN.SECOND_STAGE_EPOCHS = epochs - cfg.TRAIN.FIRST_STAGE_EPOCHS
cfg.VALID.INPUT_SIZE = 448
cfg.VALID.BATCH_SIZE = batch_size
cfg.VALID.DATA_AUG = False
cfg.TEST.INPUT_SIZE = 448
cfg.TEST.BATCH_SIZE = batch_size
cfg.TEST.DATA_AUG = False

# 根据数据集调整 NUM_CLASS
cfg.YOLO.NUM_CLASS = 2  # 设置为 2，因为您有 2 个类别
num_class = 2  # 如果需要，可定义额外变量

# ==== 使用 K-Means 生成锚框 ====
print("使用 K-Means 聚类生成锚框...")
train_annotation_file = cfg.TRAIN.ANNOT_PATH
input_size = cfg.TRAIN.INPUT_SIZE
num_clusters = 9

# 调用 utils.generate_anchors
try:
    generated_anchors = utils.generate_anchors(train_annotation_file, input_size, num_clusters)
    print(f"Anchors after K-Means clustering: {generated_anchors}")
except ValueError as e:
    print(f"锚框生成失败: {e}")
    generated_anchors = [[0, 0]] * num_clusters  # 临时解决方案

# 检查是否生成了有效的锚框
if any(anchor == [0, 0] for anchor in generated_anchors):
    print("警告: 生成的锚框包含 [0, 0]，请检查注释文件和锚框生成过程。")

# 不进行缩放，直接分配到不同尺度
anchors_per_scale = num_clusters // 3
scaled_anchors = []
for scale_idx in range(3):
    scale_anchors = generated_anchors[scale_idx * anchors_per_scale : (scale_idx + 1) * anchors_per_scale]
    scaled_anchors.append(scale_anchors)

cfg.YOLO.ANCHORS = scaled_anchors
print(f"每个尺度分配的 Anchors（未缩放）: {cfg.YOLO.ANCHORS}")

# 确保 strides 和锚框顺序与特征图顺序匹配
cfg.YOLO.STRIDES = [8, 16, 32]  # 正确的步幅顺序
cfg.YOLO.IOU_LOSS_THRESH = 0.3  # 根据需要调整
cfg.YOLO.XYSCALE = [1.2, 1.2, 1.2]  # 可选，根据需要调整

# 加载配置
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(cfg)
print("ANCHORS from load_config:", ANCHORS)
# ========== 【关键：将 ANCHORS 转为 tf.constant】 ==========
ANCHORS = [tf.constant(anchor, dtype=tf.float32) for anchor in ANCHORS]

# ==== 创建 YOLO 模型 ====
input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
feature_maps = YOLO(input_layer, NUM_CLASS, model_type, cfg.YOLO.TINY)

# 使用每个尺度的输出构建模型
model = tf.keras.Model(input_layer, feature_maps)
model.summary()

# ==== 加载权重 ====
if cfg.TRAIN.WEIGHTS_PATH is None:
    print("从头开始训练。")
else:
    if cfg.TRAIN.WEIGHTS_PATH.split(".")[-1] == "weights":
        # 指定要跳过的层（最终检测层）
        skip_layers = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']
        utils.load_weights(model, cfg.TRAIN.WEIGHTS_PATH, model_type=model_type, tiny=tiny)
    else:
        model.load_weights(cfg.TRAIN.WEIGHTS_PATH)
    print(f"从 {cfg.TRAIN.WEIGHTS_PATH} 恢复权重...")

# ==== 加载注释 ====
# 使用 utils.load_annotations 直接加载注释
print("加载注释文件...")
train_annotations = utils.load_annotations(cfg.TRAIN.ANNOT_PATH, num_classes=cfg.YOLO.NUM_CLASS, normalized=True, dataset_type='yolo')
val_annotations = utils.load_annotations(cfg.VALID.ANNOT_PATH, num_classes=cfg.YOLO.NUM_CLASS, normalized=True, dataset_type='yolo')
test_annotations = utils.load_annotations(cfg.TEST.ANNOT_PATH, num_classes=cfg.YOLO.NUM_CLASS, normalized=True, dataset_type='yolo')

print(f"已加载 {len(train_annotations)} 个训练注释。")
print(f"已加载 {len(val_annotations)} 个验证注释。")
print(f"已加载 {len(test_annotations)} 个测试注释。")

# ==== 创建带有目标生成的数据集 ====
def data_generator(annotations, input_size, strides, anchors, num_class, cfg):
    for annotation in annotations:
        image_path, bboxes = annotation
        try:
            # 读取和预处理图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, bboxes = utils.image_preprocess(image, input_size, bboxes)

            # 应用数据增强
            if cfg.TRAIN.DATA_AUG:
                image, bboxes = utils.random_horizontal_flip(image, bboxes)
                image, bboxes = utils.random_crop(image, bboxes)
                image, bboxes = utils.random_translate(image, bboxes)
                # 数据增强后再 resize 回 input_size
                image, bboxes = utils.image_resize(image, input_size, bboxes)

            # 生成目标列表
            targets = []
            for stride, anchor in zip(strides, anchors):
                # 注意：anchor 已是 tf.constant，不会与 NumPy 混用
                target = utils.generate_targets(bboxes, input_size, stride, anchor.numpy(), num_class)
                targets.append(target)

            # 将 bboxes 转换为 tf.Tensor
            bboxes_tensor = tf.convert_to_tensor(bboxes, dtype=tf.float32)

            # 将 targets 从列表转换为元组，以匹配 output_shapes 和 output_types
            yield image, tuple(targets), bboxes_tensor
        except tf.errors.InvalidArgumentError as e:
            print(f"Invalid argument for image {image_path}: {e}")
            continue
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

def create_tf_dataset(annotations, batch_size, input_size, strides, anchors, num_class, cfg, is_training=True, drop_remainder=False):
    output_types = (
        tf.float32,
        tuple(tf.float32 for _ in range(len(strides))),
        tf.float32
    )  # (图像, target列表, 边界框)

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(annotations, input_size, strides, anchors, num_class, cfg),
        output_types=output_types
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            (input_size, input_size, 3),
            tuple(
                (input_size // stride, input_size // stride, len(anchor), 5 + num_class)
                for stride, anchor in zip(strides, anchors)
            ),
            (None, 5)  # 边界框部分使用动态长度填充
        ),
        padding_values=(
            0.0,
            tuple(0.0 for _ in range(len(strides))),
            0.0
        ),
        drop_remainder=drop_remainder
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ==== 实例化数据集 ====
input_size = cfg.TRAIN.INPUT_SIZE

train_dataset = create_tf_dataset(
    annotations=train_annotations,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    input_size=input_size,
    strides=STRIDES,
    anchors=ANCHORS,
    num_class=NUM_CLASS,
    cfg=cfg,
    is_training=True,
    drop_remainder=True
)

val_dataset = create_tf_dataset(
    annotations=val_annotations,
    batch_size=cfg.VALID.BATCH_SIZE,
    input_size=input_size,
    strides=STRIDES,
    anchors=ANCHORS,
    num_class=NUM_CLASS,
    cfg=cfg,
    is_training=False,
    drop_remainder=False
)

test_dataset = create_tf_dataset(
    annotations=test_annotations,
    batch_size=cfg.TEST.BATCH_SIZE,
    input_size=input_size,
    strides=STRIDES,
    anchors=ANCHORS,
    num_class=NUM_CLASS,
    cfg=cfg,
    is_training=False,
    drop_remainder=False
)

# ==== 冻结层 ====
layer_name_prefixes = utils.load_freeze_layer(model_type=model_type, tiny=tiny)
utils.freeze_all(model, layer_name_prefixes)
print(f"已冻结的层: {layer_name_prefixes}")

# ==== 初始化优化器 ====
optimizer = Adam(learning_rate=cfg.TRAIN.LR_INIT)

# ==== TensorBoard 设置 ====
logdir = "./data/log"
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

# ==== 损失和指标历史 ====
train_total_loss_history = []
train_giou_loss_history = []
train_conf_loss_history = []
train_prob_loss_history = []

val_total_loss_history = []
val_giou_loss_history = []
val_conf_loss_history = []
val_prob_loss_history = []

val_precision_history = []
val_recall_history = []
val_f1_history = []
val_map_history = []

test_precision_history = []
test_recall_history = []
test_f1_history = []
test_map_history = []

# ==== 训练变量 ====
steps_per_epoch = len(train_annotations) // cfg.TRAIN.BATCH_SIZE
print(f"每个 epoch 的步骤数: {steps_per_epoch}")

global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
isfreeze = True

# warmup / total steps
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = (cfg.TRAIN.FIRST_STAGE_EPOCHS + cfg.TRAIN.SECOND_STAGE_EPOCHS) * steps_per_epoch
warmup_steps_tf = tf.constant(warmup_steps, dtype=tf.float32)
total_steps_tf = tf.constant(total_steps, dtype=tf.float32)
lr_init_tf = tf.constant(cfg.TRAIN.LR_INIT, dtype=tf.float32)
lr_end_tf = tf.constant(cfg.TRAIN.LR_END, dtype=tf.float32)

# ==== 验证和指标计算 ====
def calculate_TP_FP_FN(pred_boxes, gt_boxes_list, num_class, iou_threshold):
    TP = 0
    FP = 0
    FN = 0
    for img_id in range(len(gt_boxes_list)):
        gt_boxes = gt_boxes_list[img_id]
        pred = pred_boxes[img_id]

        matched_gt = []
        for p in pred:
            p_class = int(p[4])
            p_bbox = p[:4]
            max_iou = 0
            max_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if int(gt[0]) != p_class:
                    continue
                gt_bbox = gt[1:]
                iou = calculate_iou(
                    [p_bbox[0], p_bbox[1], p_bbox[2], p_bbox[3]],
                    [gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]
                )
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            if max_iou >= iou_threshold and max_gt_idx not in matched_gt:
                TP += 1
                matched_gt.append(max_gt_idx)
            else:
                FP += 1
        FN += len(gt_boxes) - len(matched_gt)
    return TP, FP, FN

def validate(model, dataset, STRIDES, ANCHORS, NUM_CLASS, IOU_LOSS_THRESH):
    total_val_loss = 0.0
    total_val_giou_loss = 0.0
    total_val_conf_loss = 0.0
    total_val_prob_loss = 0.0
    num_samples = 0

    all_pred_boxes_list = []
    all_gt_boxes_list = []

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for batch_num, (image_data, target, bboxes) in enumerate(dataset, 1):
        image_data = tf.cast(image_data, dtype=tf.float32)
        bboxes = tf.cast(bboxes, dtype=tf.float32)

        bbox_mask = tf.reduce_any(tf.not_equal(bboxes[:, :, :4], 0), axis=-1)
        valid_bboxes = tf.ragged.boolean_mask(bboxes, bbox_mask)
        valid_bboxes = utils.validate_bboxes(valid_bboxes)

        if tf.reduce_max(valid_bboxes.row_lengths()) == 0:
            print(f"验证批次 {batch_num}: 没有有效的边界框，跳过。")
            continue

        valid_bboxes_padded = valid_bboxes.to_tensor(default_value=0.0, shape=[None, None, 5])
        bboxes_mask = tf.sequence_mask(valid_bboxes.row_lengths(), maxlen=tf.shape(valid_bboxes_padded)[1])

        pred_result = model(image_data, training=False)

        # 使用 NMS 解码预测
        decoded_boxes = utils.decode_predictions(
            pred_result,
            cfg.TRAIN.INPUT_SIZE,
            STRIDES,
            ANCHORS,
            NUM_CLASS,
            XYSCALE,
            conf_threshold=0.3,
            nms_threshold=0.3
        )

        giou_loss = conf_loss = prob_loss = 0.0
        num_scales = len(pred_result)

        for i in range(num_scales):
            conv = pred_result[i]
            grid_size = tf.shape(conv)[1]

            pred = decode_train(conv, grid_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            conv = tf.reshape(conv, (tf.shape(conv)[0], grid_size, grid_size, 3, 5 + NUM_CLASS))

            loss_items = compute_loss(
                pred=pred,
                conv=conv,
                label=target[i],
                bboxes=valid_bboxes_padded,
                bboxes_mask=bboxes_mask,
                STRIDES=STRIDES,
                NUM_CLASS=NUM_CLASS,
                IOU_LOSS_THRESH=IOU_LOSS_THRESH,
                i=i,
                bbox_giou_fn=utils.bbox_giou
            )

            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        total_val_loss += total_loss.numpy()
        total_val_giou_loss += giou_loss.numpy()
        total_val_conf_loss += conf_loss.numpy()
        total_val_prob_loss += prob_loss.numpy()
        num_samples += 1

        gt_boxes_list = []
        for b in range(len(image_data)):
            gt_boxes = valid_bboxes_padded[b].numpy()
            gt_boxes = gt_boxes[gt_boxes[:, 0] != 0]
            if gt_boxes.size == 0:
                continue
            gt_converted = []
            for gt_box in gt_boxes:
                class_id, x_min, y_min, x_max, y_max = gt_box
                gt_converted.append([class_id, x_min, y_min, x_max, y_max])
            gt_boxes_list.append(np.array(gt_converted))

        for b in range(len(decoded_boxes)):
            pred_boxes = decoded_boxes[b] if decoded_boxes[b].size > 0 else np.array([])
            all_pred_boxes_list.append(pred_boxes)
            if b < len(gt_boxes_list):
                all_gt_boxes_list.append(gt_boxes_list[b])
            else:
                all_gt_boxes_list.append(np.array([]))

        TP, FP, FN = calculate_TP_FP_FN(decoded_boxes, gt_boxes_list, NUM_CLASS, IOU_LOSS_THRESH)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    mAP = utils.compute_map(all_pred_boxes_list, all_gt_boxes_list, NUM_CLASS, iou_threshold=IOU_LOSS_THRESH)
    precision, recall, f1 = utils.compute_metrics(TP=total_TP, FP=total_FP, FN=total_FN)

    print(f"验证指标 - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, mAP: {mAP:.4f}")

    avg_val_loss = total_val_loss / num_samples if num_samples > 0 else 0
    avg_val_giou_loss = total_val_giou_loss / num_samples if num_samples > 0 else 0
    avg_val_conf_loss = total_val_conf_loss / num_samples if num_samples > 0 else 0
    avg_val_prob_loss = total_val_prob_loss / num_samples if num_samples > 0 else 0

    return avg_val_loss, avg_val_giou_loss, avg_val_conf_loss, avg_val_prob_loss, precision, recall, f1, mAP

@tf.function
def train_step(image_data, targets, bboxes):
    image_data = tf.cast(image_data, dtype=tf.float32)
    targets = [tf.cast(t, dtype=tf.float32) for t in targets]
    bboxes = tf.cast(bboxes, dtype=tf.float32)

    bbox_mask = tf.reduce_any(tf.not_equal(bboxes[:, :, :4], 0), axis=-1)
    valid_bboxes = tf.ragged.boolean_mask(bboxes, bbox_mask)
    valid_bboxes = utils.validate_bboxes(valid_bboxes)
    valid_bboxes_padded = valid_bboxes.to_tensor(default_value=0.0, shape=[None, None, 5])
    bboxes_mask = tf.sequence_mask(valid_bboxes.row_lengths(), maxlen=tf.shape(valid_bboxes_padded)[1])

    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        for pred in pred_result:
            tf.debugging.check_numerics(pred, '预测包含 NaN 或 Inf')

        giou_loss = conf_loss = prob_loss = 0.0
        num_scales = len(pred_result)

        for i in range(num_scales):
            conv = pred_result[i]
            grid_size = tf.shape(conv)[1]

            pred = decode_train(conv, grid_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            conv = tf.reshape(conv, (tf.shape(conv)[0], grid_size, grid_size, 3, 5 + NUM_CLASS))

            loss_items = compute_loss(
                pred=pred,
                conv=conv,
                label=targets[i],
                bboxes=valid_bboxes_padded,
                bboxes_mask=bboxes_mask,
                STRIDES=STRIDES,
                NUM_CLASS=NUM_CLASS,
                IOU_LOSS_THRESH=cfg.YOLO.IOU_LOSS_THRESH,
                i=i,
                bbox_giou_fn=utils.bbox_giou
            )
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

    tf.debugging.assert_all_finite(total_loss, '总损失包含 NaN 或 Inf')
    tf.debugging.assert_all_finite(giou_loss, 'GIoU损失包含 NaN 或 Inf')
    tf.debugging.assert_all_finite(conf_loss, '置信度损失包含 NaN 或 Inf')
    tf.debugging.assert_all_finite(prob_loss, '类别概率损失包含 NaN 或 Inf')

    gradients = tape.gradient(total_loss, model.trainable_variables)
    for grad in gradients:
        tf.debugging.assert_all_finite(grad, '梯度包含 NaN 或 Inf')

    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    global_steps.assign_add(1)
    current_step = tf.cast(global_steps, tf.float32)

    lr = tf.cond(
        current_step < warmup_steps_tf,
        lambda: (current_step / warmup_steps_tf) * lr_init_tf,
        lambda: lr_end_tf + 0.5 * (lr_init_tf - lr_end_tf) * (
            1 + tf.cos((current_step - warmup_steps_tf) / (total_steps_tf - warmup_steps_tf) * np.pi)
        )
    )
    optimizer.lr.assign(lr)

    with writer.as_default():
        tf.summary.scalar("lr", optimizer.lr, step=global_steps)
        tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
        tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
        tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
        tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
    writer.flush()

    return giou_loss, conf_loss, prob_loss, total_loss, optimizer.lr

# ==== 主训练循环 ====
if __name__ == "__main__":
    try:
        print("创建测试图像映射...")
        test_image_dict = {}
        for annotation in test_annotations:
            image_path, gt_bboxes = annotation
            test_image_dict[image_path] = gt_bboxes

        print("测试图像映射已创建。")

        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        specified_image_paths = [
            './data/test3/images/M20220917_004823_YAHER_C8P.jpg',
            './data/test3/images/M20191015_040455_SCASS_C9P.jpg',
            './data/test3/images/M20181101_025306_SCASS_16P.jpg',
            './data/test3/images/M20191016_032529_SCASS_C6S.jpg',
        ]

        for epoch in range(epochs):
            print(f"开始第 {epoch + 1}/{epochs} 个 epoch")

            if epoch == cfg.TRAIN.FIRST_STAGE_EPOCHS and isfreeze:
                isfreeze = False
                utils.unfreeze_all(model)
                print("所有层已解冻。")

            epoch_train_total_loss = 0.0
            epoch_train_giou_loss = 0.0
            epoch_train_conf_loss = 0.0
            epoch_train_prob_loss = 0.0
            num_batches = 0

            for step, (image_data, targets, bboxes) in enumerate(train_dataset.take(steps_per_epoch), 1):
                try:
                    giou, conf, prob, total, current_lr = train_step(image_data, targets, bboxes)
                    epoch_train_total_loss += total
                    epoch_train_giou_loss += giou
                    epoch_train_conf_loss += conf
                    epoch_train_prob_loss += prob
                    num_batches += 1

                    if step % 10 == 0 or step == 1:
                        print(f"Epoch {epoch + 1}/{epochs}, Batch {step}/{steps_per_epoch}, "
                              f"Loss: GIoU={giou:.4f}, Conf={conf:.4f}, "
                              f"Prob={prob:.4f}, Total={total:.4f}, "
                              f"LR={current_lr:.6f}")
                except tf.errors.InvalidArgumentError as e:
                    print(f"训练步骤出错: {e}")
                    continue
                except Exception as e:
                    print(f"训练步骤发生意外错误: {e}")
                    continue

            if num_batches == 0:
                print(f"第 {epoch + 1} 个 epoch 中没有有效的批次。跳过验证。")
                continue

            avg_train_total_loss = epoch_train_total_loss / num_batches
            avg_train_giou_loss = epoch_train_giou_loss / num_batches
            avg_train_conf_loss = epoch_train_conf_loss / num_batches
            avg_train_prob_loss = epoch_train_prob_loss / num_batches

            train_total_loss_history.append(avg_train_total_loss.numpy())
            train_giou_loss_history.append(avg_train_giou_loss.numpy())
            train_conf_loss_history.append(avg_train_conf_loss.numpy())
            train_prob_loss_history.append(avg_train_prob_loss.numpy())

            print("开始验证...")
            (avg_val_loss, avg_val_giou_loss, avg_val_conf_loss, avg_val_prob_loss,
             precision, recall, f1, mAP) = validate(
                 model, val_dataset, STRIDES, ANCHORS, NUM_CLASS, cfg.YOLO.IOU_LOSS_THRESH
             )

            val_total_loss_history.append(avg_val_loss)
            val_giou_loss_history.append(avg_val_giou_loss)
            val_conf_loss_history.append(avg_val_conf_loss)
            val_prob_loss_history.append(avg_val_prob_loss)

            val_precision_history.append(precision)
            val_recall_history.append(recall)
            val_f1_history.append(f1)
            val_map_history.append(mAP)

            with writer.as_default():
                tf.summary.scalar('avg_train_total_loss', avg_train_total_loss, step=epoch)
                tf.summary.scalar('avg_val_total_loss', avg_val_loss, step=epoch)
                tf.summary.scalar('avg_train_giou_loss', avg_train_giou_loss, step=epoch)
                tf.summary.scalar('avg_val_giou_loss', avg_val_giou_loss, step=epoch)
                tf.summary.scalar('avg_train_conf_loss', avg_train_conf_loss, step=epoch)
                tf.summary.scalar('avg_val_conf_loss', avg_val_conf_loss, step=epoch)
                tf.summary.scalar('avg_train_prob_loss', avg_train_prob_loss, step=epoch)
                tf.summary.scalar('avg_val_prob_loss', avg_val_prob_loss, step=epoch)
                tf.summary.scalar('val_precision', precision, step=epoch)
                tf.summary.scalar('val_recall', recall, step=epoch)
                tf.summary.scalar('val_f1_score', f1, step=epoch)
                tf.summary.scalar('val_map', mAP, step=epoch)
            writer.flush()

            print(f"第 {epoch + 1} 个 epoch 训练损失: {avg_train_total_loss.numpy():.4f}")
            print(f"第 {epoch + 1} 个 epoch 验证 Precision: {precision:.4f}")
            print(f"第 {epoch + 1} 个 epoch 验证 Recall: {recall:.4f}")
            print(f"第 {epoch + 1} 个 epoch 验证 F1-Score: {f1:.4f}")
            print(f"第 {epoch + 1} 个 epoch 验证 mAP: {mAP:.4f}")

            model.save_weights(os.path.join(output_path, f"yolov4_epoch_{epoch + 1}.h5"))
            print(f"已保存第 {epoch + 1} 个 epoch 的模型权重\n")

    except KeyboardInterrupt:
        print("训练被用户中断。")
    except Exception as e:
        print(f"发生意外错误: {e}")

    # ==== 绘制并可视化 ====
    print("绘制损失曲线...")
    epochs_range = range(1, epochs + 1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 总损失
    print("Plotting loss curves...")
    try:
        plt.figure()
        plt.plot(epochs_range, train_total_loss_history, label='Train Total Loss')
        plt.plot(epochs_range, val_total_loss_history, label='Validation Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.ylim([0, 50])
        save_path = os.path.join(output_path, 'total_loss_yzoomed3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Total loss curve saved to {save_path}")
    except Exception as e:
        print(f"绘制总损失曲线时出错: {e}")

    # GIoU损失
    print("Plotting GIoU loss curve...")
    try:
        plt.figure()
        plt.plot(epochs_range, train_giou_loss_history, label='Train GIoU Loss')
        plt.plot(epochs_range, val_giou_loss_history, label='Validation GIoU Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GIoU Loss')
        plt.legend()
        save_path = os.path.join(output_path, 'giou_loss3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"GIoU loss curve saved to {save_path}")
    except Exception as e:
        print(f"绘制 GIoU 损失曲线时出错: {e}")

    # 置信度损失
    print("Plotting confidence loss curve...")
    try:
        plt.figure()
        plt.plot(epochs_range, train_conf_loss_history, label='Train Conf Loss')
        plt.plot(epochs_range, val_conf_loss_history, label='Validation Conf Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Confidence Loss')
        plt.legend()
        plt.ylim([0, 50])
        save_path = os.path.join(output_path, 'conf_loss_yzoomed3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Confidence loss curve saved to {save_path}")
    except Exception as e:
        print(f"绘制置信度损失曲线时出错: {e}")

    # 概率损失
    print("Plotting probability loss curve...")
    try:
        plt.figure()
        plt.plot(epochs_range, train_prob_loss_history, label='Train Prob Loss')



        plt.plot(epochs_range, val_prob_loss_history, label='Validation Prob Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Probability Loss')
        plt.legend()
        save_path = os.path.join(output_path, 'prob_loss3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Probability loss curve saved to {save_path}")
    except Exception as e:
        print(f"绘制类别概率损失曲线时出错: {e}")

    # 精度/召回/F1/mAP
    print("Plotting Precision, Recall, F1-Score, and mAP curves...")
    try:
        # Precision
        plt.figure()
        plt.plot(epochs_range, val_precision_history, marker='o', label='Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Validation Precision')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_path, 'precision_curve3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Precision 曲线已保存到 {save_path}")

        # Recall
        plt.figure()
        plt.plot(epochs_range, val_recall_history, marker='o', label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Validation Recall')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_path, 'recall_curve3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Recall 曲线已保存到 {save_path}")

        # F1-Score
        plt.figure()
        plt.plot(epochs_range, val_f1_history, marker='o', label='Validation F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.title('Validation F1-Score')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_path, 'f1_score_curve3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"F1-Score 曲线已保存到 {save_path}")

        # mAP
        plt.figure()
        plt.plot(epochs_range, val_map_history, marker='o', label='Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Validation mAP')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_path, 'map_curve3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"mAP 曲线已保存到 {save_path}")
    except Exception as e:
        print(f"绘制指标曲线时出错: {e}")

    # 测试集评估
    print("Starting test set evaluation...")
    try:
        avg_test_loss, avg_test_giou_loss, avg_test_conf_loss, avg_test_prob_loss, precision, recall, f1, mAP = validate(
            model, test_dataset, STRIDES, ANCHORS, NUM_CLASS, cfg.YOLO.IOU_LOSS_THRESH
        )
        print(f"Average loss on test set: {avg_test_loss}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"Test mAP: {mAP:.4f}")

        test_precision_history.append(precision)
        test_recall_history.append(recall)
        test_f1_history.append(f1)
        test_map_history.append(mAP)

        with writer.as_default():
            tf.summary.scalar('test_precision', precision, step=epochs)
            tf.summary.scalar('test_recall', recall, step=epochs)
            tf.summary.scalar('test_f1_score', f1, step=epochs)
            tf.summary.scalar('test_mAP', mAP, step=epochs)
        writer.flush()
    except Exception as e:
        print(f"Error during test set evaluation: {e}")

    # 绘制并保存所有损失曲线
    print("Plotting and saving all loss curves...")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_total_loss_history, marker='o', label='Train Total Loss')
        plt.plot(epochs_range, val_total_loss_history, marker='o', label='Validation Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Total Loss')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_path, 'loss_curve3.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved to {save_path}")
    except Exception as e:
        print(f"绘制损失曲线时出错: {e}")

    print("Starting visualization of test images...")
    try:
        ground_truths = {img_path: b for img_path, b in test_annotations}

        utils.visualize_multiple_test_images(
            model=model,
            class_names=class_names,
            image_paths=specified_image_paths,
            STRIDES=STRIDES,
            ANCHORS=ANCHORS,
            NUM_CLASS=NUM_CLASS,
            XYSCALE=XYSCALE,
            input_size=input_size,
            save_dir=os.path.join(output_path, 'visualizations7'),
            ground_truths=ground_truths
        )
    except Exception as e:
        print(f"Error during visualization of test images: {e}")
