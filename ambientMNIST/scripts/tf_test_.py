import os
import pickle
import argparse
import numpy as np
import tensorflow as tf

INPUT_SIZE = 28

param_dir = None

def get_path(file):
    global param_dir
    return os.path.join(param_dir, file)

def main(args):
    global param_dir
    param_dir = args.param_dir
    vars = {
        'input': None, 'conv1_w': None, 'conv2_w': None, 'fc1_w': None, 'fc2_w': None,
        'conv1_b': None, 'conv2_b': None, 'fc1_b': None, 'fc2_b': None,
    }
    #     'conv1_wd': None, 'conv1_md': None, 'conv1_wh': None, 
    #     'conv1_mh': None, 'conv1_wv': None, 'conv1_mv': None, 'conv1_b': None,
    #     'conv2_w': None, 'conv2_m': None, 'conv2_b': None, 'fc1_wh': None,
    #     'fc1_mh': None, 'fc1_wv': None, 'fc1_mv': None, 'fc1_b': None,
    #     'fc2_w': None, 'fc2_b': None
    # }
    
    for var in vars:
        with open(get_path(var + '.param'), 'rb') as f:
            vars[var] = pickle.load(f, encoding='latin1')
            print(var, vars[var].shape)
            
    net = {}
    for var in vars:
        net[var] = tf.compat.v1.get_variable(var, vars[var].shape, 
            initializer=tf.constant_initializer(vars[var]))

    net['input_reshape'] = tf.reshape(net['input'], [-1, INPUT_SIZE, INPUT_SIZE, 1]) 
    net['conv1d'] = tf.nn.conv2d(net['input_reshape'], net['conv1_w'], 
            [1, 1, 1, 1], padding='SAME')
    net['conv1r'] = tf.nn.relu(net['conv1d'] + net['conv1_b'])
    net['conv1max'] = tf.nn.max_pool(net['conv1r'], 
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net['conv2'] = tf.nn.conv2d(net['conv1max'], net['conv2_w'],
        [1, 1, 1, 1], padding='SAME')
    net['conv2r'] = tf.nn.relu(net['conv2'] + net['conv2_b'])
    net['conv2max'] = tf.nn.max_pool(net['conv2r'], 
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net['conv2flat'] = tf.reshape(net['conv2max'], [-1, 1600])
    net['fc1'] = tf.matmul(net['conv2flat'], net['fc1_w']) + net['fc1_b']
    net['fc1r'] = tf.nn.relu(net['fc1'])
    net['fc2'] = tf.matmul(net['fc1r'], net['fc2_w']) + net['fc2_b']

    with tf.compat.v1.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        np.set_printoptions(precision=3, linewidth=200, suppress=True)
        print(sess.run(net['fc2']).flatten().tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--param_dir',
        type=str,
        help='Parameter directory')
    args = parser.parse_args()
    tf.compat.v1.disable_eager_execution()
    main(args)