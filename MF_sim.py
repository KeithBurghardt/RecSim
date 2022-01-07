import os,ast,time,random
import pickle as pk 
import pandas as pd
import numpy as np
from scipy import sparse
import tensorflow as tf
from glob import glob
from typing import Dict
import copy





# create a sparsified matrix for data available to model
def create_Ravail(m,R_realization,fract_available,seed):
    R_avail = copy.deepcopy(R_realization)
    # create new random see of who is chosen
    if seed is not None:
        rng = np.random.seed()#seed)
    # if we hide a fraction of the user-item matrix
    if fract_available < 1.0:
       for ii,line in enumerate(R_avail):
            # hidden data is marked as "-1" elements
            line[np.random.rand(m) > fract_available] = -1
            R_avail[ii] = line
    return R_avail

# prepare matrices for fitting
def prepare_system(m,R_realization,seed,fract_available=1.0,fract_train=0.8):
    # sparsify matrix, set missing data to -1
    R_avail = create_Ravail(m,R_realization,fract_available,seed)
    R_train = []
    R_val = []
    # split data for training (80%) and testing (20%) at random
    for line in R_avail:
        line_train = line.copy()
        # remove stochastically
        remove_indices = np.random.rand(m) < fract_train
        kept_indices = np.logical_not(remove_indices)
        line_train[remove_indices] = -1
        R_train.append(line_train)

        line_val = line.copy()
        line_val[kept_indices] = -1
        R_val.append(line_val)
    R_train = np.array(R_train)+1
    R_val = np.array(R_val)+1

    R_train = sparse.coo_matrix(R_train)
    R_val = sparse.coo_matrix(R_val)
    R_train_indices = np.array(list(zip(R_train.row,R_train.col)))
    # make matrices sparse tensors
    R_train = tf.SparseTensor(indices=R_train_indices, values=R_train.data.astype(np.float64)-1, dense_shape=R_train.shape)
    R_val_indices = np.array(list(zip(R_val.row,R_val.col)))
    R_val = tf.SparseTensor(indices=R_val_indices, values=R_val.data.astype(np.float64)-1, dense_shape=R_val.shape)
    return R_avail,R_train,R_val


# approximate real data with matrix factorization
def fit_data(R_train,R_val, embeddings,U=None,V=None,wait=50,silence=False):
    indices = R_train.indices
    # use tensorflow for fast estimates of MF
    # use stochastic gradient decent for fast, memory-light model training
    if U is not None:
        U = tf.Variable(U)#convert_to_tensor(U,dtype=tf.float32)
    if U is None:
        U = tf.Variable(tf.random.uniform([R_train.shape[0], embeddings], minval=0, maxval=1/np.sqrt(embeddings)), dtype=tf.float32)
    if V is not None:
        V = tf.Variable(V)#tf.convert_to_tensor(V,dtype=tf.float32)
    if V is None:
        V = tf.Variable(tf.random.uniform([embeddings, R_train.shape[1]], minval=0, maxval=1/np.sqrt(embeddings)), dtype=tf.float32)

    optimizer = tf.optimizers.Adam()

    trainable_weights = [U, V]
    collect_vals = []
    # start with very high error
    min_val = 9999
    min_epoch = 0
    # best approximation of data so far
    best_U = tf.identity(U)
    best_V = tf.identity(V)
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = None
    # for up to 10,000 steps, train:
    for step in range(10000):
        with tf.GradientTape() as tape:
            # use SGD to fit data, stop when validation error increases
            R_prime = tf.matmul(U, V)
            # indexing the result based on the indices of A that contain a value
            num_cols = tf.shape(R_prime)[1]
            num_cols = tf.cast(num_cols, tf.int64)
            R_prime_sparse = tf.gather(
                tf.reshape(R_prime, [-1]),
                indices[:, 0] * num_cols + indices[:, 1]
            )
            # training loss
            loss = tf.reduce_sum(tf.metrics.mean_squared_error(R_prime_sparse, R_train.values))
        # gradient decent
        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        # model applied to validation data
        Rest=(U.numpy() @ V.numpy())
        Rest_val = np.array([Rest[row,col] for row,col in R_val.indices.numpy()])
        # validation loss
        val = np.mean((np.abs(Rest_val -R_val.values.numpy()))**2)
        # save if error decreasing, else stop early
        if val < min_val:
            min_val = val
            min_epoch = step
            best_U = tf.identity(U)
            best_V = tf.identity(V)
        #stop early
        if val > min_val and step-min_epoch > wait:
            break
        collect_vals.append([loss,val])
        # save important characteristics
        if step % 200 == 0 and not silence:
            print(f"Training loss at step {step}: {loss:.4f}")
            print(f"Val loss at step {step}: {val:.4f}")
    collect_vals = np.array(collect_vals)
    min_val = np.array([[np.argmin(collect_vals[:,1]),np.min(collect_vals[:,1])]])
    return best_U,best_V,min_val,collect_vals
