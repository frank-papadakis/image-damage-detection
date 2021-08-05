from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras import metrics
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from os.path import exists
import h5py
import os
from keras.models import model_from_json

def load_model():

        #global model

        json_file = open('data/model.json', 'r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("data/model.h5")
        #model._make_predict_function()
        return model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,
    beta_2=0.999)

### METRICS
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


lr_metric = get_lr_metric(optimizer)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,patience=2, min_lr=0.000000001)



### CUSTOM LOSS - FOCAL LOSS

def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        gamma=2.
        alpha=.25
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

 
checkpoint = ModelCheckpoint("model.h5py", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', save_freq='epoch')



def train_model(train_images, train_masks, validation_images, validation_masks,file):

    class DisplayCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        print(np.sum(self.model.predict(train_images[0].reshape(1,28,28,1)))) ### MUST CHANGE NAME FOR THIS

    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation = 'relu',padding='same', input_shape = (28, 28,1)))
    model.add(Conv2D(256, (5, 5),padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3),padding='same', activation = 'relu'))
    model.add(AveragePooling2D(1, (3, 3),padding='same'))
    model.add(Flatten())
    model.add(Dense(784,activation='softmax'))
    model.summary()                         

    model.compile(optimizer = optimizer,
                  loss = binary_focal_loss_fixed,
                  metrics = ['accuracy',lr_metric])
    if exists('data/model.h5'):
        model.load_weights("data/model.h5")
        
    model.fit(train_images, train_masks, epochs = 5, batch_size = 512,callbacks=[DisplayCallback(),reduce_lr,checkpoint],validation_data=(validation_images, validation_masks))
    return model
    
