from cv_bridge import CvBridge, CvBridgeError
from keras.models import model_from_json
from sensor_msgs.msg import Image

import cv2
import numpy as np
import rospy
import keras.backend as K

bridge = CvBridge()

def callback_img(data, target_size, crop_size, rootpath, save_img):
    try:
        image_type = data.encoding
        img = bridge.imgmsg_to_cv2(data, image_type)
    except CvBridgeError, e:
        print e

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = central_image_crop(img, crop_size[0], crop_size[1])

    if rootpath and save_img:
        temp = rospy.Time.now()
        cv2.imwrite("{}/{}.jpg".format(rootpath, temp), img)

    return np.asarray(img, dtype=np.float32) * np.float32(1.0/255.0)


def central_image_crop(img, crop_width, crop_heigth):
    """
    Crops the input PILLOW image centered in width and starting from the bottom
    in height.
    Arguments:
        crop_width: Width of the crop
        crop_heigth: Height of the crop
    Returns:
        Cropped image
    """
    half_the_width = img.shape[1] / 2
    img = img[(img.shape[0] - crop_heigth): img.shape[0],
              (half_the_width - (crop_width / 2)): (half_the_width + (crop_width / 2))]
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

def jsonToModel(json_model_path):
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    return model

def grad_cam(input_model, img, target_size, layer_name="add_3", scale_factor=3):
    """GradCAM method for visualizing input saliency."""

    # for steering output
    y_s = input_model.output[0][0]

    # for collision output
    y_c = input_model.output[1][0]

    # print y_s.shape, y_c.shape, img.shape

    # activation maps
    conv_output = input_model.get_layer(layer_name).output
    
    grads_s = K.gradients(y_s, conv_output)[0]
    grads_c = K.gradients(y_c, conv_output)[0]

    # print conv_output.shape, input_model.input.shape

    gradient_function = K.function([input_model.input], [conv_output, grads_s, grads_c])

    output, grad_s, grad_c = gradient_function([img])
    output, grad_s, grad_c = output[0, :], grad_s[0, :, :, :], grad_c[0, :, :, :]

    # print output.shape, grad_s.shape, grad_c.shape

    weights_s = np.mean(grad_s, axis=(0, 1))
    weights_c = np.mean(grad_c, axis=(0, 1))

    # print "weights_s, weights_c", weights_s.shape, weights_c.shape

    cam_s = np.dot(output, weights_s)
    cam_c = np.dot(output, weights_c)

    # print "cam_c.max", cam_c.max(), "cam_s.max", cam_s.max(), cam_c.shape, cam_s.shape

    # Process CAM
    cam_s = cv2.resize(cam_s, target_size, cv2.INTER_LINEAR)
    cam_s = np.maximum(cam_s, 0)
    cam_s = cam_s / (cam_s.max() + 1e-10)
    cam_s = cv2.applyColorMap(np.uint8(255 * cam_s), cv2.COLORMAP_JET)

    # print "cam_s shape after resize:", cam_s.shape

    cam_c = cv2.resize(cam_c, target_size, cv2.INTER_LINEAR)
    cam_c = np.maximum(cam_c, 0)
    cam_c = cam_c / (cam_c.max() + 1e-10)
    cam_c = cv2.applyColorMap(np.uint8(255 * cam_c), cv2.COLORMAP_JET)

    # print "cam_c shape after resize:", cam_c.shape

    final_size = (target_size[1]*scale_factor, target_size[0]*scale_factor)

    # print "final_size", final_size

    img = cv2.resize(img[0], final_size, cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.array(img*255, dtype=np.uint8)
    cam_s = cv2.resize(cam_s, final_size, cv2.INTER_LINEAR)
    cam_c = cv2.resize(cam_c, final_size, cv2.INTER_LINEAR)

    # print "img, cams, cam_c shapes before:", img.shape, cam_s.shape, cam_c.shape, type(img[0, 0, 1]), type(cam_s[0, 0, 1])

    cam_s = cv2.addWeighted(img, 0.7, cam_s, 0.3, 0)
    cam_c = cv2.addWeighted(img, 0.7, cam_c, 0.3, 0)

    # print "img, cams, cam_c shapes", img.shape, cam_s.shape, cam_c.shape

    return img, cam_s, cam_c

def np2image(np_array):
    # msg = Image()
    # msg.header.stamp = rospy.Time.now()
    # msg.encoding = "jpeg"
    # msg.data = np.array(cv2.imencode('.jpg', np_array)[1]).tostring()
    msg = bridge.cv2_to_imgmsg(np_array, encoding="bgr8")

    return msg