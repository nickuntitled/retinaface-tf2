from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from modules.models_mobilenet import RetinaFaceModel as newer_model
from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

from modules.anchor import decode_tf, prior_box_tf

flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # Paramters
    iou_th = 0.4
    score_th = 0.02

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    model_separate = newer_model(cfg, training=False, iou_th=FLAGS.iou_th, score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    model.summary()
    model_separate.summary()

    array_index = ['FPN', 'SSH_0', 'SSH_1', 'SSH_2', 'ClassHead_0', 'ClassHead_1', 
                   'ClassHead_2', 'BboxHead_0', 'BboxHead_1', 'BboxHead_2', 'LandmarkHead_0', 'LandmarkHead_1', 
                   'LandmarkHead_2']

    baselayer_weight = model.get_layer('MobileNetV2_extrator').get_weights()
    model_separate.get_layer('Mobilenet_Extractor').set_weights(baselayer_weight)
    print('Finish Transferring Weight from Base Model')

    fpnlayer = model.get_layer('FPN')
    model_separate.get_layer('FPN_conv1').set_weights(fpnlayer.output1.conv.get_weights())
    model_separate.get_layer('FPN_conv1_bn').set_weights(fpnlayer.output1.bn.get_weights())
    model_separate.get_layer('FPN_conv2').set_weights(fpnlayer.output2.conv.get_weights())
    model_separate.get_layer('FPN_conv2_bn').set_weights(fpnlayer.output2.bn.get_weights())
    model_separate.get_layer('FPN_conv3').set_weights(fpnlayer.output3.conv.get_weights())
    model_separate.get_layer('FPN_conv3_bn').set_weights(fpnlayer.output3.bn.get_weights())
    model_separate.get_layer('FPN_merge1').set_weights(fpnlayer.merge1.conv.get_weights())
    model_separate.get_layer('FPN_merge1_bn').set_weights(fpnlayer.merge1.bn.get_weights())
    model_separate.get_layer('FPN_merge2').set_weights(fpnlayer.merge2.conv.get_weights())
    model_separate.get_layer('FPN_merge2_bn').set_weights(fpnlayer.merge2.bn.get_weights())

    print('Finish transferring Weight FPN')

    for i in range(3):
        name = f'SSH_{i}'
        sshlayer = model.get_layer(name)
        model_separate.get_layer(f'{name}_conv1').set_weights(sshlayer.conv_3x3.conv.get_weights())
        model_separate.get_layer(f'{name}_conv1_bn').set_weights(sshlayer.conv_3x3.bn.get_weights())
        model_separate.get_layer(f'{name}_conv2').set_weights(sshlayer.conv_5x5_1.conv.get_weights())
        model_separate.get_layer(f'{name}_conv2_bn').set_weights(sshlayer.conv_5x5_1.bn.get_weights())
        model_separate.get_layer(f'{name}_conv3').set_weights(sshlayer.conv_5x5_2.conv.get_weights())
        model_separate.get_layer(f'{name}_conv3_bn').set_weights(sshlayer.conv_5x5_2.bn.get_weights())
        model_separate.get_layer(f'{name}_conv4').set_weights(sshlayer.conv_7x7_2.conv.get_weights())
        model_separate.get_layer(f'{name}_conv4_bn').set_weights(sshlayer.conv_7x7_2.bn.get_weights())
        model_separate.get_layer(f'{name}_conv5').set_weights(sshlayer.conv_7x7_3.conv.get_weights())
        model_separate.get_layer(f'{name}_conv5_bn').set_weights(sshlayer.conv_7x7_3.bn.get_weights())

        print('Finish transferring Weight SSH {}'.format(i))

        denselayer = model.get_layer(f'ClassHead_{i}')
        model_separate.get_layer(f'Class_{i}').set_weights(denselayer.conv.get_weights())
        print('Finish transferring Weight ClassHead')
        
        denselayer = model.get_layer(f'LandmarkHead_{i}')
        model_separate.get_layer(f'Landm_{i}').set_weights(denselayer.conv.get_weights())
        print('Finish transferring Weight LandmarkHead')

        denselayer = model.get_layer(f'BboxHead_{i}')
        model_separate.get_layer(f'BBox_{i}').set_weights(denselayer.conv.get_weights())
        print('Finish transferring Weight BboxHead')

    if not FLAGS.webcam:
        if not os.path.exists(FLAGS.img_path):
            print(f"cannot find image path from {FLAGS.img_path}")
            exit()

        print("[*] Processing on single image {}".format(FLAGS.img_path))

        img_raw = cv2.imread(FLAGS.img_path)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        if FLAGS.down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                             fy=FLAGS.down_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # run model
        inputs = img[np.newaxis, ...]
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        predictions = model_separate(x)
       
        # Bbox
        bbox_regressions = tf.concat([tf.reshape(x, [1, -1, 4]) for i,x in enumerate(predictions[0])], axis = 1)

        # Landmark
        landm_regressions = tf.concat([tf.reshape(x, [1, -1, 10]) for i,x in enumerate(predictions[1])], axis = 1)

        # Classifications
        classifications = tf.concat([tf.reshape(x, [1, -1, 2]) for i,x in enumerate(predictions[2])], axis = 1)
        classifications = tf.keras.layers.Softmax(axis=-1)(classifications)

        # Post Processing
        preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
            [bbox_regressions[0], landm_regressions[0],
             tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
             classifications[0, :, 1][..., tf.newaxis]], 1)
        priors = prior_box_tf((tf.shape(inputs)[1], tf.shape(inputs)[2]),
                              cfg['min_sizes'],  cfg['steps'], cfg['clip'])
        decode_preds = decode_tf(preds, priors, cfg['variances'])

        selected_indices = tf.image.non_max_suppression(
            boxes=decode_preds[:, :4],
            scores=decode_preds[:, -1],
            max_output_size=tf.shape(decode_preds)[0],
            iou_threshold=iou_th,
            score_threshold=score_th)

        outputs = tf.gather(decode_preds, selected_indices).numpy()
        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        # draw and save results
        save_img_path = os.path.join('out_' + os.path.basename(FLAGS.img_path))
        for prior_index in range(len(outputs)):
            draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                            img_width_raw)
            cv2.imwrite(save_img_path, img_raw)
        print(f"[*] save result at {save_img_path}")

    else:
        cam = cv2.VideoCapture(0)

        start_time = time.time()
        while True:
            _, frame = cam.read()
            if frame is None:
                print("no cam input")

            frame_height, frame_width, _ = frame.shape
            img = np.float32(frame.copy())
            if FLAGS.down_scale_factor < 1.0:
                img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                 fy=FLAGS.down_scale_factor,
                                 interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

            # run model
            outputs = model(img[np.newaxis, ...]).numpy()

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)

            # draw results
            for prior_index in range(len(outputs)):
                draw_bbox_landm(frame, outputs[prior_index], frame_height,
                                frame_width)

            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
            start_time = time.time()
            cv2.putText(frame, fps_str, (25, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
