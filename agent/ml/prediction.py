import os
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/PredNet')
import net

import threading
LOCK = threading.Lock()

class Prediction:
    def __init__(self, gpu):
        self.channels = [3,48,96,192]
        self.gpu = gpu

        #Create Model
        self.rgb0prednet = net.PredNet(200, 200, self.channels)
        self.rgb1prednet = net.PredNet(200, 200, self.channels)
        self.rgb2prednet = net.PredNet(200, 200, self.channels)
        self.rgb0model = L.Classifier(self.rgb0prednet, lossfun=mean_squared_error)
        self.rgb1model = L.Classifier(self.rgb1prednet, lossfun=mean_squared_error)
        self.rgb2model = L.Classifier(self.rgb2prednet, lossfun=mean_squared_error)
        self.rgb0model.compute_accuracy = False
        self.rgb1model.compute_accuracy = False
        self.rgb2model.compute_accuracy = False
        serializers.load_npz(os.path.abspath(os.path.dirname(__file__)) + '/models/rgb_up_300000.model', self.rgb0model)
        serializers.load_npz(os.path.abspath(os.path.dirname(__file__)) + '/models/rgb_down_300000.model', self.rgb1model)
        serializers.load_npz(os.path.abspath(os.path.dirname(__file__)) + '/models/rgb_stright_240000.model', self.rgb2model)
        # serializers.load_npz('models/rgb_stright_300000.model', rgb2model)

        self.depth0prednet = net.PredNet(32, 32, self.channels)
        self.depth1prednet = net.PredNet(32, 32, self.channels)
        self.depth2prednet = net.PredNet(32, 32, self.channels)
        self.depth0model = L.Classifier(self.depth0prednet, lossfun=mean_squared_error)
        self.depth1model = L.Classifier(self.depth1prednet, lossfun=mean_squared_error)
        self.depth2model = L.Classifier(self.depth2prednet, lossfun=mean_squared_error)
        self.depth0model.compute_accuracy = False
        self.depth1model.compute_accuracy = False
        self.depth2model.compute_accuracy = False
        serializers.load_npz(os.path.abspath(os.path.dirname(__file__)) + '/models/depth_up_300000.model', self.depth0model)
        serializers.load_npz(os.path.abspath(os.path.dirname(__file__)) + '/models/depth_down_300000.model', self.depth1model)
        serializers.load_npz(os.path.abspath(os.path.dirname(__file__)) + '/models/depth_stright_300000.model', self.depth2model)

        if self.gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device(self.gpu).use()

            self.rgb0model.to_gpu()
            self.rgb1model.to_gpu()
            self.rgb2model.to_gpu()

            self.depth0model.to_gpu()
            self.depth1model.to_gpu()
            self.depth2model.to_gpu()

    def predict(self, rgb, depth, action):
        xp = cuda.cupy if self.gpu >= 0 else np
        rgb_prednet = None
        rgb_model = None
        depth_prednet = None
        depth_model = None
        if action is 0:
            rgb_prednet = self.rgb0prednet
            rgb_model = self.rgb0model
            depth_prednet = self.depth0prednet
            depth_model = self.depth0model
        elif action is 1:
            rgb_prednet = self.rgb1prednet
            rgb_model = self.rgb1model
            depth_prednet = self.depth1prednet
            depth_model = self.depth1model
        elif action is 2:
            rgb_prednet = self.rgb2prednet
            rgb_model = self.rgb2model
            depth_prednet = self.depth2prednet
            depth_model = self.depth2model
        else:
            print 'must be action 0 or 1 or 2  rgb'
       
        rgb_prednet.reset_state()
        depth_prednet.reset_state()
        loss = 0
        batchSize = 1

        rgb_batch = np.ndarray((batchSize, self.channels[0], 200, 200), dtype=np.float32)
        rgb_y_batch = np.ndarray((batchSize, self.channels[0], 200, 200), dtype=np.float32)
        rgb_batch[0] = self._read_image(rgb, size=[200,200])
        loss += rgb_model(chainer.Variable(xp.asarray(rgb_batch)),
                      chainer.Variable(xp.asarray(rgb_y_batch)))
        loss.unchain_backward()
        loss = 0

        depth_batch = np.ndarray((batchSize, self.channels[0], 32, 32), dtype=np.float32)
        depth_y_batch = np.ndarray((batchSize, self.channels[0], 32, 32), dtype=np.float32)
        depth_batch[0] = self._read_image(depth, size=[32,32])
        loss += depth_model(chainer.Variable(xp.asarray(depth_batch)),
                      chainer.Variable(xp.asarray(depth_y_batch)))
        loss.unchain_backward()
        loss = 0
        LOCK.acquire()
        if self.gpu >= 0:
            rgb_model.to_cpu()
            depth_model.to_cpu()
        rgb_result = self._result_image(rgb_model.y.data[0].copy(), size=[200,200])
        depth_result = self._result_image(depth_model.y.data[0].copy(), size=[32,32])
        if self.gpu >= 0:
            rgb_model.to_gpu()
            depth_model.to_gpu()
        LOCK.release()
        return rgb_result, depth_result



    def _read_image(self, pil_image, size):
        image = np.asarray(pil_image).transpose(2, 0, 1)
        top = (image.shape[1]  - size[1]) / 2
        left = (image.shape[2]  - size[0]) / 2
        bottom = size[1] + top
        right = size[0] + left
        image = image[:, top:bottom, left:right].astype(np.float32)
        image /= 255
        return image

    def _result_image(self, image, size):
        image *= 255
        image = image.transpose(1, 2, 0)
        image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
        if size[0] == 200 and size[1] == 200:
            pil_img = pil_img.resize((227, 227))
        return pil_img


