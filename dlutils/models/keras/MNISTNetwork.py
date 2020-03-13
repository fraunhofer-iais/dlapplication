from DLplatform.learning.factories.kerasLearnerFactory import KerasNetwork
import numpy as np

class MNISTCNNNetwork(KerasNetwork):
    def __init__(self):
        pass
    
    def __call__(self):
        import tensorflow as tf
        from keras.models import Model
        from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
        from keras.initializers import glorot_uniform
        
        numClasses = 10
        imgRows = 28
        imgCols = 28
        inputShape = (imgRows, imgCols, 1)
        np.random.seed(42)
        tf.set_random_seed(42)
        static_initializer = glorot_uniform(seed=42)

        inp = Input(shape=inputShape)
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', 
            kernel_initializer=static_initializer)(inp)
        conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=static_initializer)(conv1)
        pool = MaxPooling2D(pool_size=(2, 2))(conv2)
        dp1 = Dropout(0.25, seed=42)(pool)
        fl = Flatten()(dp1)
        ds = Dense(128, activation='relu', kernel_initializer=static_initializer)(fl)
        dp2 = Dropout(0.5, seed=42)(ds)
        outp = Dense(numClasses, activation='softmax', kernel_initializer=static_initializer)(dp2)
        network = Model(inputs=inp, outputs=outp)
        return network
          
    def __str__(self):
        return "MNIST simple CNN"
