# coding=utf8
from data_helper import batch_iter
import embedding as emb
from model import *
import time
import os
import datetime
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import numpy as np
# from sklearn.utils import shuffle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

from input_helpers import InputHelper

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
timestamp = str(int(time.time()))
fh = logging.FileHandler('./log/log_' + timestamp + '.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

# 词向量维数
EMBEDDING_DIM = 128
# block A filter个数(model)
NUM_FILTERS_A = 50
# block B filter个数(model)
NUM_FILTER_B = 50
# 全连接层中隐藏层的单元个数
N_HIDDEN = 150
# 句子最多包含单词数(词)
SENTENCE_LENGTH = 40
# 结果分类个数(二分类后面会使用sigmod 进行优化)
NUM_CLASSES = 2
# L2正规化系数
L2_REG_LAMBDA = 1
# 训练epoch个数
NUM_EPOCHS = 8500
# mini batch大小
BATCH_SIZE = 64
# 评估周期(单位step)
EVALUATE_EVERY = 100
# 模型存档周期
CHECKPOINT_EVERY = 2000
# 优化器学习率
LR = 1e-3

# llow device soft device placement
ALLOW_SOFT_PLACEMENT = True
# Log placement of ops on devices
LOG_DEVICE_PLACEMENT = False

# 原始训练文件
TRAINING_FILES_RAW = './train_data/atec_nlp_sim_train.csv'
# 验证集比例
DEV_PERCENT = 1

# 自己训练的word2vec模型
WORD2VEC_MODEL_SELF = './word2vec_model.bin'

# word2vec模型（采用已训练好的中文模型）
# WORD2VEC_MODEL = '../word2vecmodel/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
WORD2VEC_MODEL = WORD2VEC_MODEL_SELF
# 　模型格式为bin
WORD2VEC_FORMAT = 'bin'

# 卷积filter大小
filter_size = [1, 2, SENTENCE_LENGTH]

inpH = InputHelper()
# 训练自己的word2vec模型
# inpH.gen_word2vec(TRAINING_FILES_RAW, WORD2VEC_MODEL_SELF, EMBEDDING_DIM)
# exit(0)

train_set, dev_set, vocab_processor, sum_no_of_batches = inpH.getDataSets(TRAINING_FILES_RAW, SENTENCE_LENGTH,
                                                                          DEV_PERCENT,
                                                                          BATCH_SIZE)

# print(type(vocab_processor.vocabulary_._mapping))
# for index, k in enumerate(vocab_processor.vocabulary_._mapping):
#     print('vocab-{}, {}:{}'.format(index, k,vocab_processor.vocabulary_._mapping[k]))
#     print('======:{}'.format(vocab_processor.vocabulary_.reverse(vocab_processor.vocabulary_._mapping[k])))

# origin_sentence='为啥我花呗叫话费都交不了'
# print(origin_sentence)
# sentence_list=list(vocab_processor.transform(np.asarray([origin_sentence])))
#
# print(sentence_list)
# sentence=[]
# for idx in sentence_list[0]:
#     word=vocab_processor.vocabulary_.reverse(idx)
#     sentence.append(word)
#     print(word)
#     for k, v in vocab_processor.vocabulary_._mapping.items():
#         if word==k:
#             print(k, v)
# print ('/'.join(sentence))
#
#
# exit(0)


Xtrain = [train_set[0], train_set[1]]

# for item in Xtrain:
#     print (item)
#
#
# print (type(Xtrain))
# print (len(Xtrain))
# print (type(Xtrain[0]))
# print (len(Xtrain[0]))
# print (Xtrain[0].shape)
# print (Xtrain[0].dtype)
# exit(0)

ytrain = train_set[2]

# for item in ytrain:
#     print (item)
# print (type(ytrain))
# print (len(ytrain))
# print (ytrain.shape)
# exit(0)

# Xtrain[0], Xtrain[1], ytrain = shuffle(Xtrain[0], Xtrain[1], ytrain)

# print ('======after shuffle======')
# print (type(Xtrain[0]))
# print (Xtrain[0].shape)
# print (Xtrain[0].dtype)
# exit(0)

Xtest = [dev_set[0], dev_set[1]]
ytest = dev_set[2]
# Xtest[0], Xtest[1], ytest = shuffle(Xtest[0], Xtest[1], ytest)


with tf.Session() as sess:
    input_1 = tf.placeholder(tf.int32, [None, SENTENCE_LENGTH], name="input_x1")
    input_2 = tf.placeholder(tf.int32, [None, SENTENCE_LENGTH], name="input_x2")
    input_3 = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="input_y")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # 加载word2vec
    inpH.loadW2V(WORD2VEC_MODEL, WORD2VEC_FORMAT)
    # initial matrix with random uniform
    initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), EMBEDDING_DIM)).astype(np.float32)
    # print (initW)
    # print (type(initW))
    # exit(0)

    # print(initW)
    # sys.exit(0)

    # load any vectors from the word2vec
    print("initializing initW with pre-trained word2vec embeddings")
    for index, w in enumerate(vocab_processor.vocabulary_._mapping):
        # print('vocab-{}:{}'.format(index, w))

        arr = []
        if w in inpH.pre_emb:
            arr = inpH.pre_emb[w]
            # print('=====arr-{},{}'.format(index, arr))
            idx = vocab_processor.vocabulary_.get(w)
            initW[idx] = np.asarray(arr).astype(np.float32)
        else:
            pass
            # print ('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

            # 不使用词向量
            # arr=[]
            # idx = vocab_processor.vocabulary_.get(w)
            # arr.append(idx)
            # initW[idx] = np.asarray(arr).astype(np.float32)

    print("Done assigning intiW. len=" + str(len(initW)))
    # exit(0)

    # for idx, value in enumerate(initW):
    #     print(idx, value)
    # sys.exit(0)

    # sess.run(siameseModel.W.assign(initW))

    with tf.name_scope("embendding"):
        s0_embed = tf.nn.embedding_lookup(initW, input_1)
        s1_embed = tf.nn.embedding_lookup(initW, input_2)

    with tf.name_scope("reshape"):
        input_x1 = tf.reshape(s0_embed, [-1, SENTENCE_LENGTH, EMBEDDING_DIM, 1])
        input_x2 = tf.reshape(s1_embed, [-1, SENTENCE_LENGTH, EMBEDDING_DIM, 1])
        input_y = tf.reshape(input_3, [-1, NUM_CLASSES])

    # sent1_unstack = tf.unstack(input_x1, axis=1)
    # sent2_unstack = tf.unstack(input_x2, axis=1)
    # D = []
    # for i in range(len(sent1_unstack)):
    #     d = []
    #     for j in range(len(sent2_unstack)):
    #         dis = compute_cosine_distance(sent1_unstack[i], sent2_unstack[j])
    #         d.append(dis)
    #     D.append(d)
    # D = tf.reshape(D, [-1, len(sent1_unstack), len(sent2_unstack), 1])
    # A = [tf.nn.softmax(tf.expand_dims(tf.reduce_sum(D, axis=i), 2)) for i in [2, 1]]
    #
    # print A[1]
    # print A[1] * input_x2
    # atten_embed = tf.concat([input_x2, A[1] * input_x2], 2)

    setence_model = MPCNN_Layer(NUM_CLASSES, EMBEDDING_DIM, filter_size,
                                [NUM_FILTERS_A, NUM_FILTER_B], N_HIDDEN,
                                input_x1, input_x2, input_y, dropout_keep_prob, L2_REG_LAMBDA)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    setence_model.similarity_measure_layer()
    optimizer = tf.train.AdamOptimizer(LR)
    grads_and_vars = optimizer.compute_gradients(setence_model.loss)
    train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    # print("Writing to {}\n".format(out_dir))
    #
    loss_summary = tf.summary.scalar("loss", setence_model.loss)
    acc_summary = tf.summary.scalar("accuracy", setence_model.accuracy)
    f1_summary = tf.summary.scalar('f1', setence_model.f1)
    #
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    #
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))


    def train(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        # for index, sentence in enumerate(x1_batch):
        #     word_list1=[]
        #     word_list2=[]
        #     y=y_batch[index]
        #     for idx in x1_batch[index]:
        #         word_list1.append(vocab_processor.vocabulary_.reverse(idx))
        #     for idx in x2_batch[index]:
        #         word_list2.append(vocab_processor.vocabulary_.reverse(idx))
        #
        #     # print(''.join(word_list1),'\t',''.join(word_list2),'\t',y)
        #     print('==========={}=============='.format(index))
        #     print('/'.join(word_list1))
        #     print ('/'.join(word_list2))
        #     print(y)
        # exit(0)

        # feed_dict = {
        #     input_1: x1_batch,
        #     input_2: x2_batch,
        #     input_3: y_batch,
        #     dropout_keep_prob: 0.5
        # }

        feed_dict = {
            input_1: x1_batch,
            input_2: x2_batch,
            input_3: y_batch,
            dropout_keep_prob: 0.8  # drpout实际上并未生效
        }
        _, step, summaries, batch_loss, accuracy, f1, y_out = sess.run(
            [train_step, global_step, train_summary_op, setence_model.loss, setence_model.accuracy, setence_model.f1,
             setence_model.output],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logger.info("{}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(time_str, step, batch_loss, accuracy, f1))
        # logger.info('y_out= {}'.format(y_out))
        train_summary_writer.add_summary(summaries, step)


    def dev_step(x1_batch, x2_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            input_1: x1_batch,
            input_2: x2_batch,
            input_3: y_batch,
            dropout_keep_prob: 1
        }
        step, summaries, batch_loss, accuracy, f1 = sess.run(
            [global_step, dev_summary_op, setence_model.loss, setence_model.accuracy, setence_model.f1],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        dev_summary_writer.add_summary(summaries, step)
        # if writer:
        #     writer.add_summary(summaries, step)

        return batch_loss, accuracy


    sess.run(tf.global_variables_initializer())
    batches = batch_iter(list(zip(Xtrain[0], Xtrain[1], ytrain)), BATCH_SIZE, NUM_EPOCHS)
    for batch in batches:
        x1_batch, x2_batch, y_batch = zip(*batch)

        train(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % EVALUATE_EVERY == 0:
            total_dev_loss = 0.0
            total_dev_accuracy = 0.0

            logger.info("\nEvaluation:")
            dev_batches = batch_iter(list(zip(Xtest[0], Xtest[1], ytest)), BATCH_SIZE, 1)
            for dev_batch in dev_batches:
                x1_dev_batch, x2_dev_batch, y_dev_batch = zip(*dev_batch)
                dev_loss, dev_accuracy = dev_step(x1_dev_batch, x2_dev_batch, y_dev_batch)
                total_dev_loss += dev_loss
                total_dev_accuracy += dev_accuracy
            total_dev_accuracy = total_dev_accuracy / (len(ytest) / BATCH_SIZE)
            logger.info("dev_loss {:g}, dev_acc {:g}, num_dev_batches {:g}".format(total_dev_loss, total_dev_accuracy,
                                                                                   len(ytest) / BATCH_SIZE))
            # train_summary_writer.add_summary(summaries)

        if current_step % CHECKPOINT_EVERY == 0:
            saver.save(sess, checkpoint_prefix, global_step=current_step)

    logger.info("Optimization Finished!")
