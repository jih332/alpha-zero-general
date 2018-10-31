import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import tensorflow as tf

args = dotdict({
    'architecture': 'CNN',
    'lr': 0.001,
    'dropout': 0.5,
    'survival_decay': 0.2,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 32,
})

class NNetWrapper(NeuralNet):
    def __init__(self,game):
        if args.architecture == 'CNN':
            self.nnet = CNN(game, args)
        elif args.architecture[0] == 'Sdepth_ResNet':
            self.nnet = Sdepth_ResNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.sess = tf.Session(graph=self.nnet.graph)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            examples_inds = np.random.permutation(range(len(examples)))
            start_ind = 0
            end_ind = args.batch_size

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = examples_inds[start_ind: end_ind]
                start_ind = end_ind
                end_ind += args.batch_size
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # predict and compute gradient and do SGD step
                input_dict = {self.nnet.input_boards: boards, self.nnet.target_pis: pis, self.nnet.target_vs: vs, self.nnet.dropout: args.dropout, self.nnet.isTraining: True}

                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                self.sess.run(self.nnet.train_step, feed_dict=input_dict)
                pi_loss, v_loss = self.sess.run([self.nnet.loss_pi, self.nnet.loss_v], feed_dict=input_dict)
                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        prob, v = self.sess.run([self.nnet.prob, self.nnet.v], feed_dict={self.nnet.input_boards: board, self.nnet.dropout: 0, self.nnet.isTraining: False})

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return prob[0], v[0]

    def save_checkpoint(self, folder='./checkpoints/', filename='ckpt_0'):
        if isinstance(filename, tuple):
            filename = filename[0] + str(filename[1])
        filename = filename + '.pth.tar'
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Make new directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:            
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)
        print("model saved {}".format(filepath))

    def load_checkpoint(self, folder='./checkpoints/', filename='ckpt_0'):
        if isinstance(filename, tuple):
            filename = filename[0] + str(filename[1])
        filename = filename + '.pth.tar'
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'.meta'):
            raise("No saved model found in {}".format(filepath))
        print("load model {}".format(filename))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)


class CNN():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense
        Softmax = tf.nn.softmax

         # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")

            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])          # batch_size  x board_x x board_y x 1
            conv1 = Relu(BatchNormalization(self.conv_3(x_image, args.num_channels), axis=3, training=self.isTraining))
            conv2 = Relu(BatchNormalization(self.conv_3(conv1, args.num_channels*2), axis=3, training=self.isTraining))
            conv3 = Relu(BatchNormalization(self.conv_3(conv2, args.num_channels*4, padding='VALID'), axis=3, training=self.isTraining))
            features = Relu(BatchNormalization(self.conv_3(conv3, args.num_channels*8, padding='VALID'), axis=3, training=self.isTraining))

            pi_conv = Relu(BatchNormalization(self.conv_3(features, args.num_channels*2), axis = 3, training = self.isTraining))
            v_conv = Relu(BatchNormalization(self.conv_3(features, args.num_channels), axis = 3, training = self.isTraining))

            pi_flat = tf.reshape(pi_conv, [-1, args.num_channels * 2 * (self.board_x - 4) * (self.board_y - 4)])
            v_flat = tf.reshape(v_conv, [-1, args.num_channels * (self.board_x - 4) * (self.board_y - 4)])

            pi_fc = Dropout(Relu(BatchNormalization(Dense(pi_flat, 256), axis = 1, training = self.isTraining)), rate = self.dropout)
            v_fc = Dropout(Relu(BatchNormalization(Dense(v_flat, 128), axis = 1, training = self.isTraining)), rate = self.dropout)

            self.pi = Dense(pi_fc, self.action_size)
            self.prob = Softmax(self.pi)
            self.v = Tanh(Dense(v_fc, 1))                  # batch_size x self.action_size
                                                             
            self.calculate_loss()

    def conv_3(self, x, out_channels, strides = [1, 1], padding = 'SAME'):
        return tf.layers.conv2d(x, out_channels, kernel_size = [3, 3], strides = strides, padding = padding)

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)



class Sdepth_ResNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.bottleneck = True if args.architecture[1] == 'bottleneck' else False
        self.projection = True if args.architecture[2] == 'projection' else False

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense
        Softmax = tf.nn.softmax

         # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")

            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])          # batch_size  x board_x x board_y x 1
            conv1 = Relu(BatchNormalization(self.conv_3(x_image, args.num_channels*2), axis=3, training=self.isTraining))
            resblock1 = self.res_block(conv1, int(args.num_channels/2), args.num_channels*2, 1.0, scope = 'resblock1')
            resblock2 = self.res_block(resblock1, args.num_channels, args.num_channels*4, 1.0 - args.survival_decay, scope = 'resblock2')
            features = self.res_block(resblock2, args.num_channels, args.num_channels*4, 1.0 - 2 * args.survival_decay, scope = 'resblock3')

            pi_conv = Relu(BatchNormalization(self.conv_3(features, args.num_channels*2), axis = 3, training = self.isTraining))
            v_conv = Relu(BatchNormalization(self.conv_3(features, args.num_channels), axis = 3, training = self.isTraining))

            pi_flat = tf.reshape(pi_conv, [-1, args.num_channels * 2 * (self.board_x - 4) * (self.board_y - 4)])
            v_flat = tf.reshape(v_conv, [-1, args.num_channels * (self.board_x - 4) * (self.board_y - 4)])

            pi_fc = Dropout(Relu(BatchNormalization(Dense(pi_flat, 256), axis = 1, training = self.isTraining)), rate = self.dropout)
            v_fc = Dropout(Relu(BatchNormalization(Dense(v_flat, 128), axis = 1, training = self.isTraining)), rate = self.dropout)

            self.pi = Dense(pi_fc, self.action_size)
            self.prob = Softmax(self.pi)
            self.v = Tanh(Dense(v_fc, 1))                  # batch_size x self.action_size
                                                             
            self.calculate_loss()

    def conv2d(self, x, kernel_shape, scope, strides = [1, 1, 1, 1], padding = 'SAME', use_relu = True):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            weights = tf.get_variable(name = 'weights', shape = kernel_shape)
            bias = tf.get_variable(name = 'bias', shape = kernel_shape[3])
            conv = tf.nn.bias_add(tf.nn.conv2d(x, weights, strides, padding), bias)
            batch_norm = tf.contrib.layers.batch_norm(conv, updates_collections = None, is_training = self.isTraining)
            output = tf.nn.relu(batch_norm) if use_relu else batch_norm
            return output

    def res_block(self, x, bottleneck_channels, out_channels, survival_rate, scope):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            in_channels = x.get_shape()[3].value
            if in_channels == out_channels:
                strides = [1, 1, 1, 1]
                res = x
            else:
                strides = [1, 2, 2, 1]
                if self.projection:
                    res = self.conv2d(x = x, 
                                      kernel_shape = [1, 1, in_channels, out_channels], 
                                      scope = 'shortcut', 
                                      strides = strides)
                else:
                    res = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, out_channels - in_channels]])
                    res = tf.contrib.layers.max_pool2d(res, kernel_size = [1, 1], stride = 2, padding = 'SAME')
            survival_rate = tf.constant(survival_rate)

            def originblock():
                block = self.conv2d(x = x,
                                    kernel_shape = [3, 3, in_channels, bottleneck_channels],
                                    scope = 'originblock_in',
                                    strides = strides)
                block_out = self.conv2d(x = block,
                                    kernel_shape = [3, 3, bottleneck_channels, out_channels],
                                    scope = 'originblock_out',
                                    use_relu = False)
                return block_out

            def bottleneck():
                bottleneck_1 = self.conv2d(x = x, 
                                           kernel_shape = [1, 1, in_channels, bottleneck_channels], 
                                           scope = 'bottleneck_1')
                bottleneck_3 = self.conv2d(x = bottleneck_1, 
                                           kernel_shape = [3, 3, bottleneck_channels, bottleneck_channels], 
                                           scope = 'bottleneck_3', 
                                           strides = strides)
                bottleneck_out = self.conv2d(x = bottleneck_3, 
                                             kernel_shape = [1, 1, bottleneck_channels, out_channels], 
                                             scope = 'bottleneck_out', 
                                             use_relu = False)
                return bottleneck_out

            def training():
                def thru_block():
                    output = bottleneck() if self.bottleneck else originblock()
                    output = tf.nn.relu(tf.add(output, res))
                    return output

                def skip_block():
                    output = tf.nn.relu(res)
                    return output

                survive = tf.random_uniform(shape = [], minval = 0., maxval = 1., dtype = tf.float32)
                survive = tf.less(survive, survival_rate)
                return tf.cond(survive, thru_block, skip_block)
            
            def testing():
                output = bottleneck() if self.bottleneck else originblock()
                output = tf.multiply(output, survival_rate)
                output = tf.nn.relu(tf.add(output, res))
                return output

            return tf.cond(self.isTraining, training, testing)


    def conv_3(self, x, out_channels, strides = [1, 1], padding = 'SAME'):
        return tf.layers.conv2d(x, out_channels, kernel_size = [3, 3], strides = strides, padding = padding)


    def conv_1(self, x, out_channels, strides = [1, 1], padding = 'SAME'):
        return tf.layers.conv2d(x, out_channels, kernel_size = [1, 1], strides = strides, padding = padding)


    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)
