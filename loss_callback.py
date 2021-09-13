import tensorflow.keras.callbacks as Callback

class RealTrainLossCallback(Callback):
    # def on_train_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Starting training; got log keys: {}".format(keys))

    # def on_train_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop training; got log keys: {}".format(keys))

    # def on_epoch_begin(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        val_acc = logs['val_acc']
        history = self.model.predict(self.x_train)
        f = open("loss_log_trainvsval.txt","a")
        f.write('epoch'+str(epoch))
        f.write('val_loss,'+str(val_loss)+'\ntraining_loss,'+str( history.history['loss'])+'\n')
        f.write('val_acc,'+str(val_acc)+'\ntraining_acc,'+str( history.history['acc'])+'\n')
        f.close()
        print("End epoch {} of training; real training loss: {}".format(epoch, history.history['loss']))

    # def on_test_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start testing; got log keys: {}".format(keys))

    # def on_test_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop testing; got log keys: {}".format(keys))

    # def on_predict_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start predicting; got log keys: {}".format(keys))

    # def on_predict_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop predicting; got log keys: {}".format(keys))

    # def on_train_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_train_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    # def on_test_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_test_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    # def on_predict_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_predict_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

