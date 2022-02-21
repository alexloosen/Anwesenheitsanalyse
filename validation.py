def plotHistoryAccuracy(history):
    mplt.plot(history.history['accuracy'])
    mplt.plot(history.history['val_accuracy'])
    mplt.title('model accuracy')
    mplt.ylabel('accuracy')
    mplt.xlabel('epoch')
    mplt.legend(['train', 'val'], loc='lower right')
    return mplt

def plotHistoryLoss(history):
    mplt.plot(history.history['loss'])
    mplt.plot(history.history['val_loss'])
    mplt.title('model loss')
    mplt.ylabel('loss')
    mplt.xlabel('epoch')
    mplt.legend(['train', 'val'], loc='upper right')
    return mplt
