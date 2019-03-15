#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:25:12 2018

@author: amir

see: http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
v2: attempt to prevent cluttering tensorflow graph and speedup save and load

"""
import time
import os
import pickle

from typing import List, Tuple
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.models import load_model as keras_load_model

from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
import tensorflow as tf


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

def create_nn(n_inputs, n_cells, bidirectional, device='cpu', name='nn') -> Sequential:
    '''
    creates a recursive neural network
    arguments
    =========
            device: gpu or cpu
    '''
    model = Sequential()
#    model.add(layers.GRU(32, dropout=.2, recurrent_dropout=.2, input_shape=(None, n_inputs) ))
#    model.add(layers.LSTM(8, dropout=.2, recurrent_dropout=.2, input_shape=(None, n_inputs) ))

#    model.add(layers.LSTM(n_cells, dropout=.2, recurrent_dropout=.2,
#                          input_shape=(None, n_inputs),
#                          kernel_constraint=maxnorm(3) ))
    
#    if bidirectional:
#        model.add(layers.Bidirectional(
#                  layers.LSTM(n_cells, 
#                          input_shape=(None, n_inputs) ), merge_mode='concat' ))
#    else:
#        model.add(layers.LSTM(n_cells, 
#                          input_shape=(None, n_inputs) ))
                          
    if bidirectional:
        if device == 'cpu':
            model.add(layers.Bidirectional(
                      layers.LSTM(n_cells),
                          input_shape=(None, n_inputs), merge_mode='concat', name=name))  #  input_shape is for Bidirectional, otherwise we get error when loading the saved network
        elif device == 'gpu':
            model.add(layers.Bidirectional(
                      layers.CuDNNLSTM(n_cells),
                          input_shape=(None, n_inputs), merge_mode='concat', name=name ))  #  input_shape is for Bidirectional, otherwise we get error when loading the saved network
        else:
            raise ValueError('device has to be cpu or gpu')    
    else:
        if device == 'cpu':
            model.add(layers.LSTM(n_cells, 
                          input_shape=(None, n_inputs), name=name ))
        elif device == 'gpu':
            model.add(layers.CuDNNLSTM(n_cells, 
                          input_shape=(None, n_inputs), name=name ))
        else:
            raise ValueError('device has to be cpu or gpu')    
            
    model.add(layers.Dense(1))

    #model.compile(optimizer=RMSprop(), loss='mean_squared_error') # 'mae'
    #model.compile(optimizer=RMSprop(), loss='mae') # 'mae'
    #model.compile(optimizer=Adam(), loss='mae') # 'mae'
    model.compile(optimizer=Adam(), loss='mean_squared_error') # 'mae'
    return model


def one_boost_step(model, X, y, r, lr) -> Tuple[float]:
    """
    parameters:
        r: residual
        lr: learning rate
    """
    r2 = model.predict(X)  # model is trained to output residual
    y2 = y - r + lr * r2  # y2: new approximation of y
    r = y - y2
    return r, y2

def one_boost_step_test():
    import numpy as np
    class Model():
        def __init__(self, output):
            self.output = output # the model always outputs 'output'
        def predict(self, X):
            return np.ones((X.shape[0], X.shape[-1]))* self.output
    
    X = np.array([1,2])
    X = X.reshape(1, 2, 1)
    y = np.array([1]).reshape(1,1)
    model = Model(.25-.05)
    r = np.array([.25]).reshape(1,1)
    lr = .1
    rr, yy2 = one_boost_step(model, X, y, r, lr)
    np.testing.assert_approx_equal(rr[0,0], .23)
    np.testing.assert_approx_equal(yy2[0,0], .77)
    
"""
from the reference above:
Parameters and init
As model_selection.GridSearchCV uses set_params to apply parameter setting to
 estimators, it is essential that calling set_params has the same effect as 
 setting parameters using the __init__ method. The easiest and recommended way
 to accomplish this is to not do any parameter validation in __init__. All 
 logic behind estimator parameters, like translating string arguments into 
 functions, should be done in fit.
Also it is expected that parameters with trailing _ are not to be set
 inside the __init__ method. All and only the public attributes set by 
 fit have a trailing _. As a result the existence of parameters with 
 trailing _ is used to check if the estimator has been fitted.
"""

#TODO: add epochs and earlystop=True/False as parameter?
# TODO: receive the model as parameter and omit   bidirectional=True, n_cells=2
class Nngboost(BaseEstimator, RegressorMixin):
    def __init__(self, bidirectional=True, n_inputs=10, n_cells=2, n_estimators=50, lr=.1,
                 epochs=1000, batch_size=128, device='cpu'):
        """
        paramaters:
            lr: learning rate of the boosting method
        """
        self.bidirectional = bidirectional
        self.n_inputs = n_inputs
        self.n_cells = n_cells
        self.n_estimators = n_estimators
        self.lr = lr
        self.lr_decay_const = 70
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
    def fit(self, X, y, X_val, y_val):
        callback = [EarlyStopping(patience=10, verbose=1)]
        self.nn_:List[Sequential] = []
        self.graph_:List[tf.Graph] = []
        self.sess_:List[tf.Session] = []
        r2_val_list = []
        r, r_val = y, y_val
        for i in range(self.n_estimators):
            print('~~~~~~~~~~~~~~~~~~~')
            print(f'estimator number {i}:')
            print('learning rate is : ' , self.lr_updated(i))
            self.graph_.append(tf.Graph())
            self.sess_.append(tf.Session(graph=self.graph_[-1]))
            #with self.graph_[-1].as_default():
            #with tf.Session(graph=self.graph_[-1]):
            with self.graph_[i].as_default():
              with self.sess_[i].as_default():   
                self.nn_.append(create_nn(self.n_inputs, self.n_cells,
                                          self.bidirectional, self.device, f'nn{i}'))
                history = self.nn_[-1].fit(x=X, y=r, epochs=self.epochs, batch_size=self.batch_size, 
                            validation_data=(X_val, r_val), callbacks=callback,
                            verbose=0)
                r, y2 = one_boost_step(self.nn_[-1], X, y, r, self.lr_updated(i))
                r_val, y2_val = one_boost_step(self.nn_[-1], X_val, y_val, r_val, self.lr_updated(i))
            
            print('r2 train: ', r2_score(y, y2))
            r2_val_list.append(r2_score(y_val, y2_val))
            print('r2 validation: ', r2_val_list[-1])
        self.n_estimators_best_ = np.argmax(r2_val_list) + 1
        print('best number of estimators based on validation r2: ', self.n_estimators_best_)
        print('best validation r2: ', r2_val_list[self.n_estimators_best_-1])
        return self
        
    def lr_updated(self, itr):
        return self.lr * 2**(-itr / self.lr_decay_const)
    
    def predict(self, X):
        check_is_fitted(self, ['nn_', 'n_estimators_best_'])
        #X = check_array(X)
        if len(X.shape) != 3 :
            raise ValueError('the dimension of input must be 3 ')
        #with tf.Session(graph=self.graph_[0]):
        with self.graph_[0].as_default():
          with self.sess_[0].as_default():
            y = self.nn_[0].predict(X)
        for i in range(1, self.n_estimators_best_):
            #with  tf.Session(graph=self.graph_[i]):
            with self.graph_[i].as_default():
              with self.sess_[i].as_default():
                y+=  self.lr_updated(i) * self.nn_[i].predict(X)
        return y
    def save(self, model_name):
        if not os.path.isdir(model_name):
            os.mkdir(model_name)
        for i in range(len(self.nn_)):
            #with tf.Session(graph=self.graph_[i]):
            with self.graph_[i].as_default():
              with self.sess_[i].as_default():
                self.nn_[i].save(model_name + '/' + str(i) + '.h5')
                print(f'nn {i} saved')
        parameters = {}
        for k, v in self.__dict__.items():
            if k != 'nn_' and k != 'sess_' and k != 'graph_':
                parameters.update({k:v})
        with open(model_name+'/parameters.pkl', 'wb') as f:
            pickle.dump(parameters, f)

def load_model(model_name):
    model = Nngboost()
    with open(model_name+'/parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
    p = {}
    for k, v in model.__dict__.items():
        if k != 'nn_':
            p.update({k:v})
    s = set(p)
    s.add('n_estimators_best_')
    if s != set(parameters):
        print('parameters as set is: ', set(parameters))
        raise ValueError('the saved nngboost model has different attributes than the current version of the model class')
    for k, v in parameters.items():
        model.__dict__[k] = v
    print('model is loaded with the parameters:', parameters)
    model.nn_ = []
    model.graph_ = []
    model.sess_ = []
    for i in range(parameters['n_estimators']):
        t1 = time.time()
        model.graph_.append(tf.Graph())
        model.sess_.append(tf.Session(graph=model.graph_[-1]))
        with model.graph_[i].as_default():
           with model.sess_[i].as_default():
        #with tf.Session(graph=self.graph_[i]):
            model.nn_.append(keras_load_model(model_name + '/' + str(i) + '.h5'))
            print(f'nn {i} loaded.', 'time taken:', time.time()-t1)
    print('loading the pretrained model finished')
    return model
