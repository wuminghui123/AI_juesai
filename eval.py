#读取存储的图，可运行

import tensorflow as tf
import os
import scipy.io as sio 
import numpy as np
import math
import time
import struct
import numpy as np
import datetime

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# 配置1：选择在昇腾AI处理器上执行推理
custom_op.parameter_map["use_off_line"].b = True

# 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

# 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
custom_op.parameter_map["graph_run_mode"].i = 0

# 配置4：关闭remapping
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF




def generatorXY(batch, H,SNR,Pilotnum):
    input_labels = []
    input_samples = []
    input_H = []
    for row in range(0, batch):
        #mode = np.random.randint(0, 3)
        mode = 0
        SNRdb = np.random.randint(0, 5)+SNR
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_H.append(HH)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_H = np.asarray(input_H)
    return batch_y, batch_x

def decode(Y,Pilotnum):
    N = Y.shape[0]
    Y_mat = np.transpose(np.reshape(Y,[N,256,8]),[0,2,1])
    Y = np.reshape(Y,[N,1,256,8])

    Y_t = np.zeros([N,1,256,8])
    Y_t[:,0,:,0] = Y_mat[:,0,:]
    Y_t[:,0,:,1] = Y_mat[:,1,:]
    Y_t[:,0,:,2] = Y_mat[:,4,:]
    Y_t[:,0,:,3] = Y_mat[:,5,:]

    Y_t[:,0,:,4] = Y_mat[:,2,:]
    Y_t[:,0,:,5] = Y_mat[:,3,:]
    Y_t[:,0,:,6] = Y_mat[:,6,:]
    Y_t[:,0,:,7] = Y_mat[:,7,:]
    Y = Y_t
    # produces the expected result.
    x_2 = tf.placeholder("float", shape=[None, 784], name="input")
    y__2 = tf.placeholder("float", [None, 10])

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = "./pb_model/model_"+str(Pilotnum)+".best.pb"
        #sess.graph.add_to_collection("input", mnist.test.images)

        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session(config=config) as sess:

            tf.initialize_all_variables().run()
            if Pilotnum == 8:
                input_x = sess.graph.get_tensor_by_name("input_7:0")
                print(input_x)
                output = sess.graph.get_tensor_by_name("flatten_13/Reshape:0")
                print(output)
            else:
                input_x = sess.graph.get_tensor_by_name("input_8:0")
                print(input_x)
                output = sess.graph.get_tensor_by_name("flatten_15/Reshape:0")
                print(output)
            y_conv_2 = sess.run(output,{input_x:Y})
        
            X_pre = np.array(np.floor(y_conv_2 + 0.5), dtype=np.bool)
    return X_pre
N = 10000

# Y, X = generatorXY(N, H,8,8)
# X_pre = decode(Y,8)
# acc = np.sum(X_pre == X)/N/1024
# print('Pilot8 : The accuracy is',acc)

# Y, X = generatorXY(N, H,8,32)
# X_pre = decode(Y,32)
# acc = np.sum(X_pre == X)/N/1024
# print('Pilot32 : The accuracy is',acc)

Y_1 = np.loadtxt('./data/Y_1.csv', delimiter=',')
Y_2 = np.loadtxt('./data/Y_2.csv', delimiter=',')


 
start = datetime.datetime.now()
X_pre_1 = decode(Y_1,32)
time0 =  datetime.datetime.now()-start
print('done32 ','time:',time0)

start = datetime.datetime.now()
X_pre_2 = decode(Y_2,8)
time0 =  datetime.datetime.now()-start
print('done8 ','time:',time0)
X_pre_1.tofile('./data/X_pre_1.bin')
X_pre_2.tofile('./data/X_pre_2.bin')
