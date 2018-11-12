import sys
caffe_root = "/home/alfa/Documents/caffe/"
sys.path.insert(0, caffe_root + "python")
import caffe

net = caffe.Net("./models/mp_from_conv/dw2a_train_val.prototxt", "./pretrain/bvlc_reference_caffenet.caffemodel", caffe.TEST)

print net.params

layer_list = ["conv5", "fc6", "fc7"]

for layer in layer_list:
    dst = layer + "_2"
    net.params[dst][0].data[...] = net.params[layer][0].data[...]
    net.params[dst][1].data[...] = net.params[layer][1].data[...]

net.save("./pretrain/pretrain_from_conv.caffemodel")

