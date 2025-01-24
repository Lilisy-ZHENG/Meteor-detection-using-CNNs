import tensorflow as tf
import core.common as common

def csp_block(input_data, filters, num_blocks, name):
    """
    定义 CSP Block，用于 CSPDarknet53 主干网络。
    """
    # 将输入分为两部分
    split1 = tf.keras.layers.Lambda(lambda x: x[:, :, :, : filters // 2],
                                    name=f"{name}_split1")(input_data)
    split2 = tf.keras.layers.Lambda(lambda x: x[:, :, :, filters // 2 :],
                                    name=f"{name}_split2")(input_data)

    # 将 split1 通过多个残差块
    residual = split1
    for i in range(num_blocks):
        residual = common.residual_block(
            residual,
            input_channels=filters // 2,   # 或其他值
            output_channels1=filters // 4, # 或其他值
            output_channels2=filters // 2, # 或其他值
            name=f"{name}_residual_{i+1}"
        )

    # 将经过残差块处理的 residual 与 split2 拼接
    concatenated = tf.keras.layers.Concatenate(name=f"{name}_concat")([residual, split2])

    # 通过一个 1x1 卷积层
    output = common.convolutional(
        concatenated, (1, 1, filters, filters), name=f"{name}_conv"
    )
    return output


def CSPDarknetLite(input_data):
    """
    一个更轻量的 CSPDarknet 主干网络示例。
    减少卷积通道以及残差块数量，适合在小目标/小数据集上快速实验。
    """
    # ------------------ Stage 1 ------------------
    # 减少初始卷积通道数
    input_data = common.convolutional(input_data, (3, 3, 3, 16), name="darklite_conv1")

    # 下采样
    input_data = common.convolutional(
        input_data, (3, 3, 16, 32), downsample=True, name="darklite_conv2"
    )
    # CSP Block 1: 1 个残差块 (较少)
    input_data = csp_block(input_data, filters=32, num_blocks=1, name="darklite_block1")

    # ------------------ Stage 2 ------------------
    # 下采样
    input_data = common.convolutional(
        input_data, (3, 3, 32, 64), downsample=True, name="darklite_conv3"
    )
    # CSP Block 2: 1~2 个残差块
    input_data = csp_block(input_data, filters=64, num_blocks=2, name="darklite_block2")

    # ------------------ Stage 3 ------------------
    # 下采样
    input_data = common.convolutional(
        input_data, (3, 3, 64, 128), downsample=True, name="darklite_conv4"
    )
    # CSP Block 3
    input_data = csp_block(input_data, filters=128, num_blocks=2, name="darklite_block3")
    route1 = input_data  # 作为第一分支输出

    # ------------------ Stage 4 ------------------
    # 下采样
    input_data = common.convolutional(
        input_data, (3, 3, 128, 256), downsample=True, name="darklite_conv5"
    )
    # CSP Block 4
    input_data = csp_block(input_data, filters=256, num_blocks=2, name="darklite_block4")
    route2 = input_data  # 作为第二分支输出

    # ------------------ Stage 5 ------------------
    # 下采样
    input_data = common.convolutional(
        input_data, (3, 3, 256, 512), downsample=True, name="darklite_conv6"
    )
    # CSP Block 5
    input_data = csp_block(input_data, filters=512, num_blocks=1, name="darklite_block5")
    backbone_output = input_data

    return route1, route2, backbone_output


def CSPDarknet53(input_data):
    """
    原版的 CSPDarknet53 主干网络。
    (适合在 YOLOv4 / YOLOv4-CSP 中使用)
    """
    # Stage 1
    input_data = common.convolutional(input_data, (3, 3, 3, 32), name="darknet53_conv1")

    # Stage 2
    input_data = common.convolutional(
        input_data, (3, 3, 32, 64), downsample=True, name="darknet53_conv2"
    )
    # CSP Block 1: 1 个残差块
    input_data = csp_block(
        input_data, filters=64, num_blocks=1, name="csp_darknet53_block1"
    )

    # Stage 3
    input_data = common.convolutional(
        input_data, (3, 3, 64, 128), downsample=True, name="darknet53_conv3"
    )
    # CSP Block 2: 2 个残差块
    input_data = csp_block(
        input_data, filters=128, num_blocks=2, name="csp_darknet53_block2"
    )

    # Stage 4
    input_data = common.convolutional(
        input_data, (3, 3, 128, 256), downsample=True, name="darknet53_conv4"
    )
    # CSP Block 3: 8 个残差块
    input_data = csp_block(
        input_data, filters=256, num_blocks=8, name="csp_darknet53_block3"
    )
    route1 = input_data  # 输出 1

    # Stage 5
    input_data = common.convolutional(
        input_data, (3, 3, 256, 512), downsample=True, name="darknet53_conv5"
    )
    # CSP Block 4: 8 个残差块
    input_data = csp_block(
        input_data, filters=512, num_blocks=8, name="csp_darknet53_block4"
    )
    route2 = input_data  # 输出 2

    # Stage 6
    input_data = common.convolutional(
        input_data, (3, 3, 512, 1024), downsample=True, name="darknet53_conv6"
    )
    # CSP Block 5: 4 个残差块
    input_data = csp_block(
        input_data, filters=1024, num_blocks=4, name="csp_darknet53_block5"
    )
    backbone_output = input_data  # 主干最终输出

    return route1, route2, backbone_output


def CSPDarknetDeeper(input_data):
    """
    一个更深的 CSPDarknet 主干网络示例。
    增加残差块数量，以获取更深层次的特征表达。
    """
    # ------------------ Stage 1 ------------------
    input_data = common.convolutional(input_data, (3, 3, 3, 32), name="darkdeep_conv1")

    # Stage 2
    input_data = common.convolutional(
        input_data, (3, 3, 32, 64), downsample=True, name="darkdeep_conv2"
    )
    # CSP Block 1: 2 个残差块 (比原版多一些)
    input_data = csp_block(
        input_data, filters=64, num_blocks=2, name="csp_darkdeep_block1"
    )

    # Stage 3
    input_data = common.convolutional(
        input_data, (3, 3, 64, 128), downsample=True, name="darkdeep_conv3"
    )
    # CSP Block 2: 4 个残差块 (比原版 2 更多)
    input_data = csp_block(
        input_data, filters=128, num_blocks=4, name="csp_darkdeep_block2"
    )

    # Stage 4
    input_data = common.convolutional(
        input_data, (3, 3, 128, 256), downsample=True, name="darkdeep_conv4"
    )
    # CSP Block 3: 16 个残差块 (比原版 8 更多)
    input_data = csp_block(
        input_data, filters=256, num_blocks=16, name="csp_darkdeep_block3"
    )
    route1 = input_data

    # Stage 5
    input_data = common.convolutional(
        input_data, (3, 3, 256, 512), downsample=True, name="darkdeep_conv5"
    )
    # CSP Block 4: 16 个残差块
    input_data = csp_block(
        input_data, filters=512, num_blocks=16, name="csp_darkdeep_block4"
    )
    route2 = input_data

    # Stage 6
    input_data = common.convolutional(
        input_data, (3, 3, 512, 1024), downsample=True, name="darkdeep_conv6"
    )
    # CSP Block 5: 8 个残差块 (比原版 4 更多)
    input_data = csp_block(
        input_data, filters=1024, num_blocks=8, name="csp_darkdeep_block5"
    )
    backbone_output = input_data

    return route1, route2, backbone_output


