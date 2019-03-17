# gboostnn
## Gradient Boosting of neural network regressors

I have developed this code within the project MVControl (https://iktderzukunft.at/en/projects/mv-control.php), funded by Austrian Research Promoion agency (FFG).

The code implements gradient boosting using neural networks (here: bidirectional LSTM, that could be easily replace with any othre network architecture) defined by Keras with Tensorflow backend.

The class is made on top of Scikit-learn BaseEstimator and it is tried to be compatible with it. So the model, can be created, fitted and asked for prediction as a normal scikit regressor.

One of the diffuculties of using NN (Keras, with a tensorflow backend) is that adding consequtive estimators gradually slows down the training and prediction. In other words, because of cluttering the grpah model by consequtive NNs, each epoch takes longer for later NNs. This code is optimized from this point of view. such that adding estimators doesn't slow down the process. This is addressed by adding each new network  to a new graph. This problem also is taken care when saving and loading the trained model. So saving and loading the model is pretty fast and just linearly increases by the size of the model (i.e. the number of estimators)

## TODO list

TODO add usage example

TODO add unit test

TODO add feature: for more flexibility to pass the base network as an argument (if it still remains compatible with scikit-learn).
