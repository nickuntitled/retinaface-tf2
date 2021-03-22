import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Input, Conv2D, ReLU, LeakyReLU, Add, Concatenate, UpSampling2D, Reshape, Multiply
from modules.anchor import decode_tf, prior_box_tf


def _regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal()


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)

#  Custom Resize Layer
class ResizeLayer(tf.keras.layers.Layer):
    """Conv + BN + Act"""
    def __init__(self, imgtarget, name = "ResizeLayer", **kwargs):
        super(ResizeLayer, self).__init__(name=name, **kwargs)
        self.up_h, self.up_w = tf.shape(imgtarget)[1], tf.shape(imgtarget)[2]

    def call(self, x):
        return tf.image.resize(x, [self.up_h, self.up_w], method = 'nearest')

def Backbone(x, backbone_type='Mobilenet', use_pretrain=True):
    """Backbone Model"""
    extractor = MobileNetV2(
                input_shape=x.shape[1:], include_top=False, weights='imagenet')
    pick_layer1 = 54  # [80, 80, 32]
    pick_layer2 = 116  # [40, 40, 96]
    pick_layer3 = 143  # [20, 20, 160]
            #preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    #extractor = ResNet50(
    #        input_shape=x.shape[1:], include_top=False, weights='imagenet')
    #pick_layer1 = 80  # [80, 80, 512]
    #pick_layer2 = 142  # [40, 40, 1024]
    #pick_layer3 = 174  # [20, 20, 2048]

    return Model(extractor.input,
                     (extractor.layers[pick_layer1].output,
                      extractor.layers[pick_layer2].output,
                      extractor.layers[pick_layer3].output),
                     name='Mobilenet_Extractor')(x)

    #return backbone

def ConvUnit(x, f, k, s, wd, name, **kwargs):
    conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                           kernel_initializer=_kernel_init(),
                           kernel_regularizer=_regularizer(wd),
                           use_bias=False, name=name)
    bn = BatchNormalization(name=f'{name}_bn')
    act_fn = LeakyReLU(0.1)

    return act_fn(bn(conv(x)))

def ConvUnit_identity(x, f, k, s, wd, name, **kwargs):
    conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                           kernel_initializer=_kernel_init(),
                           kernel_regularizer=_regularizer(wd),
                           use_bias=False, name=name)
    bn = BatchNormalization(name=f'{name}_bn')

    return bn(conv(x))

def FPN(input1, input2, input3, out_ch, wd, name, **kwargs):
    output1 = ConvUnit(input1, f=out_ch, k=1, s=1, wd=wd ,name=f'{name}_conv1')  # [80, 80, out_ch]
    output2 = ConvUnit(input2, f=out_ch, k=1, s=1, wd=wd, name=f'{name}_conv2')  # [40, 40, out_ch]
    output3 = ConvUnit(input3, f=out_ch, k=1, s=1, wd=wd, name=f'{name}_conv3')  # [20, 20, out_ch]

    #up_h, up_w = tf.shape(output2)[1], tf.shape(output2)[2]
    #up3 = tf.image.resize(output3, [up_h, up_w], method='nearest')
    up3 = UpSampling2D(size=(2,2), interpolation='nearest')(output3)
    output2 = Add()([output2, up3])
    output2 = ConvUnit(output2, f=out_ch, k=3, s=1, wd=wd, name=f'{name}_merge2')

    #up_h, up_w = tf.shape(output1)[1], tf.shape(output1)[2]
    #up2 = tf.image.resize(output2, [up_h, up_w], method='nearest')
    up2 = UpSampling2D(size=(2,2), interpolation='nearest')(output2)
    output1 = Add()([output1, up2])
    output1 = ConvUnit(output1, f=out_ch, k=3, s=1, wd=wd, name=f'{name}_merge1' )

    return output1, output2, output3

def SSH(x, out_ch, wd, name, **kwargs):
    conv_3x3 = ConvUnit_identity(x, f=out_ch // 2, k=3, s=1, wd=wd, name=f'{name}_conv1')

    conv_5x5_1 = ConvUnit(x, f=out_ch // 4, k=3, s=1, wd=wd, name=f'{name}_conv2')
    conv_5x5 = ConvUnit_identity(conv_5x5_1, f=out_ch // 4, k=3, s=1, wd=wd, name=f'{name}_conv3')

    conv_7x7_2 = ConvUnit(conv_5x5_1, f=out_ch // 4, k=3, s=1, wd=wd, name=f'{name}_conv4')
    conv_7x7 = ConvUnit_identity(conv_7x7_2, f=out_ch // 4, k=3, s=1, wd=wd, name=f'{name}_conv5')

    output = Concatenate(axis=3)([conv_3x3, conv_5x5, conv_7x7]) #, axis=3)
    output = ReLU()(output)

    return output

def BboxHead(x, num_anchor, wd, name):
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    x = Conv2D(filters=num_anchor * 4, kernel_size=1, strides=1, name=name)(x)

    return x #Reshape([-1, Multiply()([h,w,num_anchor]), 4])(x)

def LandmarkHead(x, num_anchor, wd, name):
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    x = Conv2D(filters=num_anchor * 10, kernel_size=1, strides=1, name=name)(x)

    return x #Reshape([-1, Multiply()([h,w,num_anchor]), 10])(x)

def ClassHead(x, num_anchor, wd, name):
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    x = Conv2D(filters=num_anchor * 2, kernel_size=1, strides=1, name=name)(x)

    return x #Reshape([-1, Multiply()([h,w,num_anchor]), 2])(x)

def RetinaFaceModel(cfg, training=False, iou_th=0.4, score_th=0.02,
                    name='RetinaFaceModel'):
    """Retina Face Model"""
    input_size = cfg['input_size'] if training else None
    wd = cfg['weights_decay']
    out_ch = cfg['out_channel']
    num_anchor = len(cfg['min_sizes'][0])
    backbone_type = cfg['backbone_type']

    # define model
    x = inputs = Input([input_size, input_size, 3], name='input_image')

    #if training:
    backbone_output1, backbone_output2, backbone_output3 = Backbone(x, backbone_type=backbone_type)

    fpn = FPN(backbone_output1, backbone_output2, backbone_output3 , out_ch=out_ch, wd=wd, name='FPN')

    features = [SSH(f, out_ch=out_ch, wd=wd, name=f'SSH_{i}')
                for i, f in enumerate(fpn)]

    print('Bbox Regression')
    bbox_regressions = [BboxHead(f, num_anchor, wd=wd, name=f'BBox_{i}')
         for i, f in enumerate(features)]

    print('Landm Regression')
    landm_regressions = [LandmarkHead(f, num_anchor, wd=wd, name=f'Landm_{i}') 
        for i, f in enumerate(features)]

    print('Classificaitons')
    classifications = [ClassHead(f, num_anchor, wd=wd, name=f'Class_{i}')
         for i, f in enumerate(features)]

    #classifications = tf.keras.layers.Softmax(axis=-1)(classifications)

    #if training:
    out = (bbox_regressions, landm_regressions, classifications, features)
    
    #else:
        # only for batch size 1
    #    preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
    #        [bbox_regressions[0], landm_regressions[0],
    #         tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
    #         classifications[0, :, 1][..., tf.newaxis]], 1)
    #    priors = prior_box_tf((tf.shape(inputs)[1], tf.shape(inputs)[2]),
    #                          cfg['min_sizes'],  cfg['steps'], cfg['clip'])
    #    decode_preds = decode_tf(preds, priors, cfg['variances'])

    #    selected_indices = tf.image.non_max_suppression(
    #        boxes=decode_preds[:, :4],
    #        scores=decode_preds[:, -1],
    #        max_output_size=tf.shape(decode_preds)[0],
    #        iou_threshold=iou_th,
    #        score_threshold=score_th)

    #    out = tf.gather(decode_preds, selected_indices)

    return Model(inputs, out, name=name)
