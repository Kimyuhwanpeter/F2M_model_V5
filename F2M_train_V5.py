# -*- coding:utf-8 -*-
from F2M_model_V5 import *
from random import shuffle

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 416,
                           
                           "batch_size": 1,
                           
                           "epochs": 200,

                           "lr": 0.0002,

                           "train": True,

                           "pre_checkpoint": False,

                           "pre_checkpoint_path": "",

                           ""
                           
                           "tr_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/AFAD/All/female_40_63",
                           
                           "tr_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_40_63_16_39/train/female_40_63_train.txt",
                           
                           "re_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_16_39",
                           
                           "re_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_40_63_16_39/train/male_16_39_train.txt"})

ge_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
de_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def input_func(img_zip, lab_zip):

    tr_img = tf.io.read_file(img_zip[0])
    tr_img = tf.image.decode_jpeg(tr_img, 3)
    tr_img = tf.image.resize(tr_img, [FLAGS.img_size, FLAGS.img_size]) / 127.5 - 1.
    re_img = tf.io.read_file(img_zip[1])
    re_img = tf.image.decode_jpeg(re_img, 3)
    re_img = tf.image.resize(re_img, [FLAGS.img_size, FLAGS.img_size]) / 127.5 - 1.

    tr_lab = lab_zip[0]
    re_lab = lab_zip[1]

    return tr_img, re_img, tr_lab, re_lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(inp_images, ref_images, ge_model, de_model):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        o_images = run_model(ge_model, inp_images, True)    # 이제 이 부분들을 수정해주어야한다.
        d_real = run_model(de_model, [inp_images, ref_images], True)
        d_fake = run_model(de_model, [inp_images, o_images])

        g_gan_loss = tf.reduce_mean((d_fake - tf.ones_like(d_fake))**2)
        g_target_loss = tf.reduce_mean(tf.abs(o_images - ref_images))
        g_loss = 10 * g_target_loss + g_gan_loss

        d_loss = ( tf.reduce_mean((d_real - tf.ones_like(d_real))**2) + tf.reduce_mean((d_fake - tf.zeros_like(d_fake))**2) ) / 2.

    g_grads = g_tape.gradient(g_loss, ge_model.trainable_variables)
    d_grads = d_tape.gradient(d_loss, de_model.trainable_variables)

    ge_optim.apply_gradients(zip(g_grads, ge_model.trainable_variables))
    de_optim.apply_gradients(zip(d_grads, de_model.trainable_variables))

    return g_loss, d_loss

def main():
    ge_model = new_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    de_model = new_discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    ge_model.summary()
    de_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(ge_model=ge_model, de_model=de_model,
                                   ge_optim=ge_optim, de_optim=de_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!")

    if FLAGS.train:
        count = 0

        tr_img = np.loadtxt(FLAGS.tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + "/" + data for data in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        re_img = np.loadtxt(FLAGS.re_txt_path, dtype="<U100", skiprows=0, usecols=0)
        re_img = [FLAGS.re_img_path + "/" + data for data in re_img]
        re_lab = np.loadtxt(FLAGS.re_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(FLAGS.epochs):

            img_zip = list(zip(tr_img, re_img))
            shuffle(img_zip)
            img_zip = np.array(img_zip)

            lab_zip = list(zip(tr_lab, tr_lab))
            shuffle(lab_zip)
            lab_zip = np.array(lab_zip)

            data_generator = tf.data.Dataset.from_tensor_slices((img_zip, lab_zip))
            data_generator = data_generator.shuffle(len(tr_img))
            data_generator = data_generator.map(input_func)
            data_generator = data_generator.batch(FLAGS.batch_size)
            data_generator = data_generator.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(data_generator)
            tr_idx = len(tr_img) // FLAGS.batch_size
            for step in range(tr_idx):
                inp_images, ref_images, inp_labels, ref_labels = next(tr_iter)

                g_loss, d_loss = cal_loss(inp_images, ref_images, ge_model, de_model)

                print(g_loss, d_loss)

                if count % 100 == 0:
                    o_images = run_model(ge_model, inp_images, False)

                    plt.imsave("C:/Users/Yuhwan/Pictures/img/fake_{}.jpg".format(count), o_images[0] * 0.5 + 0.5)
                    plt.imsave("C:/Users/Yuhwan/Pictures/img/real_{}.jpg".format(count), inp_images[0] * 0.5 + 0.5)

                count += 1

if __name__ == "__main__":
    main()