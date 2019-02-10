#!/usr/bin/env python
import rospy
from dronet_perception.msg import CNN_out
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty
import numpy as np
import cv2
import utils

from keras import backend as K

TEST_PHASE = 0


class Dronet(object):
    def __init__(self,
                 json_model_path,
                 weights_path, target_size=(200, 200),
                 crop_size=(150, 150),
                 imgs_rootpath="../models",
                 get_saliency_maps=False):

        self.pub = rospy.Publisher("cnn_predictions", CNN_out, queue_size=5)
        self.feedthrough_sub = rospy.Subscriber(
            "state_change", Bool, self.callback_feedthrough, queue_size=1)
        self.land_sub = rospy.Subscriber(
            "land", Empty, self.callback_land, queue_size=1)

        self.gradcam_st_pub = None
        self.gradcam_col_pub = None
        self.gradcam_nn_pub = rospy.Publisher("neural_ip", Image, queue_size=1)

        self.get_saliency_maps = get_saliency_maps
        if self.get_saliency_maps:
            self.gradcam_st_pub = rospy.Publisher(
                "steering_am", Image, queue_size=1)
            self.gradcam_col_pub = rospy.Publisher(
                "collision_am", Image, queue_size=1)

        self.use_network_out = True
        self.imgs_rootpath = imgs_rootpath

        # Set keras utils
        K.set_learning_phase(TEST_PHASE)

        # Load json and create model
        model = utils.jsonToModel(json_model_path)
        # Load weights
        model.load_weights(weights_path)
        print("Loaded model from {}".format(weights_path))

        model.compile(loss='mse', optimizer='sgd')
        self.model = model
        self.target_size = target_size
        self.crop_size = crop_size

    def callback_feedthrough(self, data):
        self.use_network_out = data.data

    def callback_land(self, data):
        self.use_network_out = False

    def run(self):
        while not rospy.is_shutdown():
            msg = CNN_out()
            msg.header.stamp = rospy.Time.now()
            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message("camera", Image, timeout=10)
                except:
                    pass

            if self.use_network_out:
                print("Publishing commands!")
            else:
                print("NOT Publishing commands!")

            cv_image = utils.callback_img(data, self.target_size, self.crop_size,
                                          self.imgs_rootpath, self.use_network_out)
            outs = self.model.predict_on_batch(cv_image[None])

            if self.get_saliency_maps:
                img, cam_s, cam_c = utils.grad_cam(
                    self.model, cv_image[None], self.crop_size, layer_name="add_3", scale_factor=2)

                cam_s_msg = utils.np2image(cam_s)
                cam_c_msg = utils.np2image(cam_c)
                img_msg = utils.np2image(img)

                self.gradcam_nn_pub.publish(img_msg)
                self.gradcam_col_pub.publish(cam_c_msg)
                self.gradcam_st_pub.publish(cam_s_msg)
            else:
                img = np.array(cv_image*255, dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img_msg = utils.np2image(img)
                self.gradcam_nn_pub.publish(img_msg)

            steer, coll = outs[0][0], outs[1][0]
            msg.steering_angle = steer*30
            msg.collision_prob = coll
            self.pub.publish(msg)
