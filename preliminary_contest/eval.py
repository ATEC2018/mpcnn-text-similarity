#! /usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
import sys

# Parameters
# ==================================================
EVAL_FILE = sys.argv[1]  # 待评估文件
OUTPUT_FILE = sys.argv[2]  # 评估后输出文件

print (EVAL_FILE)
print (OUTPUT_FILE)

# Eval Parameters
BATCH_SIZE = 64  # 批大小
VOCAB_FILE = './vocab/vocab'  # 训练使使用的词表
MODEL = './models/model-40000'  # 加载训练模型
ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = False

# 语句最多长度(包含多少个词)
MAX_DOCUMENT_LENGTH = 40

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test, x2_test = inpH.getTestDataSet(EVAL_FILE, VOCAB_FILE, MAX_DOCUMENT_LENGTH)

# for index, _ in enumerate(x1_test):
#     print(index, x1_test[index], x2_test[index])

# print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = MODEL
# print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=ALLOW_SOFT_PLACEMENT,
        log_device_placement=LOG_DEVICE_PLACEMENT)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        print('start restore model.........')
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)
        print('restore model finished')

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        batches = inpH.batch_iter(list(zip(x1_test, x2_test)), BATCH_SIZE, 1, shuffle=False)
        all_d = []

        print('start batch iteration............')
        for db in batches:
            x1_dev_b, x2_dev_b = zip(*db)
            batch_sim = sess.run(sim, {input_x1: x1_dev_b, input_x2: x2_dev_b, dropout_keep_prob: 1.0})
            # print(batch_sim)
            # print(type(batch_sim))
            # print(len(batch_sim))

            all_d = np.concatenate([all_d, batch_sim])

        f_output = open(OUTPUT_FILE, 'a')
        index = 1
        predic_value = 0
        for item in all_d:
            if item > 0:
                predic_value = 1
            else:
                predic_value = 0
            f_output.write('{}\t{}\n'.format(index, predic_value))
            index += 1

        print ('eval finished!')
