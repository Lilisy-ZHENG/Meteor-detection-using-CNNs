# core/common.py

import tensorflow as tf

from tensorflow.keras import regularizers

def convolutional(input_data, filters_shape, downsample=False, activate=True, bn=True, kernel_regularizer=None, name=None):
    """
    定义卷积层，包含可选的下采样、批归一化、激活以及L2正则化。

    Args:
        input_data (tf.Tensor): 输入张量
        filters_shape (tuple): 卷积核形状，例如 (3, 3, 3, 32)
        downsample (bool): 是否进行下采样（步幅为2）
        activate (bool): 是否应用激活函数
        bn (bool): 是否应用批归一化
        kernel_regularizer (tf.keras.regularizers.Regularizer): 正则化方法，默认为None
        name (str): 层名称

    Returns:
        tf.Tensor: 输出张量
    """
    if downsample:
        # stride 2 conv 使用 valid padding
        input_data = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_data)
        padding = 'valid'
        strides = (2,2)
    else:
        padding = 'same'
        strides = (1,1)
    
    # Apply Conv2D with optional L2 regularization
    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=(filters_shape[0], filters_shape[1]),
        strides=strides,
        padding=padding,
        use_bias=False if bn else True,
        kernel_initializer=tf.keras.initializers.HeNormal(),
        kernel_regularizer=kernel_regularizer,  # Added kernel_regularizer here
        name=f"{name}_conv" if name else None
    )(input_data)
    
    # Apply Batch Normalization if required
    if bn:
        conv = CustomBatchNormalization(epsilon=1e-5, name=f"{name}_bn" if name else None)(conv)
    
    # Apply LeakyReLU activation if required
    if activate:
        conv = tf.keras.layers.LeakyReLU(alpha=0.1, name=f"{name}_leakyrelu" if name else None)(conv)
    
    return conv
def residual_block(input_data, input_channels, output_channels1, output_channels2, name=None):
    """
    定义残差块：Conv -> Conv -> [Add + Activation]
    
    Args:
        input_data (tf.Tensor): 输入张量
        input_channels (int): 输入张量的通道数
        output_channels1 (int): 第一个卷积的输出通道数
        output_channels2 (int): 第二个卷积的输出通道数
        name (str): 块名称(可选)，方便网络调试时查看

    Returns:
        tf.Tensor: 残差块输出张量
    """
    # ---- 分支1: 主干卷积分支 ----
    conv = convolutional(
        input_data,
        (1, 1, input_channels, output_channels1),
        downsample=False,
        activate=True,
        bn=True,
        name=f"{name}_conv1" if name else None
    )
    conv = convolutional(
        conv,
        (3, 3, output_channels1, output_channels2),
        downsample=False,
        activate=False,  # 可以先不激活，最后Add完再激活
        bn=True,
        name=f"{name}_conv2" if name else None
    )

    # ---- 分支2: 捷径(Shortcut)分支。若通道数不同，就做1x1卷积对齐 ----
    route = input_data
    if input_channels != output_channels2:
        route = convolutional(
            route,
            (1, 1, input_channels, output_channels2),
            downsample=False,
            activate=False,
            bn=True,
            name=f"{name}_route_adjust" if name else None
        )

    # ---- Add + Activation ----
    output = tf.keras.layers.Add(name=f"{name}_add" if name else None)([route, conv])
    # 这里也可换成别的激活函数，如 tf.nn.relu 等
    output = tf.nn.leaky_relu(output, alpha=0.1, name=f"{name}_leaky" if name else None)

    return output


def upsample(input_data, name=None):
    """
    定义上采样层，使用双线性插值。

    Args:
        input_data (tf.Tensor): 输入张量
        name (str): 层名称

    Returns:
        tf.Tensor: 上采样后的输出张量
    """
    return tf.keras.layers.UpSampling2D(
        size=(2, 2), 
        interpolation='bilinear', 
        name=name
    )(input_data)

class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, epsilon=1e-5, **kwargs):
        """
        自定义批归一化层，增加数值稳定性。

        Args:
            epsilon (float): 防止除以零的小常数
            **kwargs: 其他参数
        """
        super(CustomBatchNormalization, self).__init__(epsilon=epsilon, **kwargs)

    def call(self, inputs, training=False):
        """
        重写 call 方法，确保在训练时正确使用。

        Args:
            inputs (tf.Tensor): 输入张量
            training (bool): 是否在训练模式

        Returns:
            tf.Tensor: 批归一化后的输出
        """
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super(CustomBatchNormalization, self).call(inputs, training)


