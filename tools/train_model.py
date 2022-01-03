import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def define_model_1():
    # Creating Sequential model
    # using sequential model for training
    model = Sequential()
    # 1st layer and taking input in this of shape 100x100x3 ->  100 x 100 pixles and 3 channels
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    # maxpooling will take highest value from a filter of 2*2 shape
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # it will prevent overfitting by making it hard for the model to idenify the images
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    # last layer predicts 17 labels
    model.add(Dense(17, activation="softmax"))
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    model.summary()
    # Visualising the model
    # displaying the model
    keras.utils.plot_model(model, "model.png", show_shapes=True)
    return model


def define_model_2(x_train):
    # Define model 2.
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(17, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    model.summary()
    # Visualising the model
    # displaying the model
    keras.utils.plot_model(model, "model.png", show_shapes=True)
    return model


def define_model_3(x_train):
    # Define model 3.
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=17, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    model.summary()
    # Visualising the model
    # displaying the model
    keras.utils.plot_model(model, "model.png", show_shapes=True)
    return model


def train_model_1(X_train, y_train, X_test, y_test,  disp_mod_acc=True, disp_mod_loss=True, predict=True):
    # training the model
    model = define_model_1()
    history = model.fit(
        X_train,
        y_train,
        batch_size=50,
        epochs=200,
        validation_split=0.2,
        shuffle=True
    )
    model.save('C:\Users\Stipe\PycharmProjects\PhotoMathTask\models\model.h5')
    # Visualising the outcome
    # displaying the model accuracy
    if disp_mod_acc:
        plt.plot(history.history['accuracy'], label='train', color="red")
        plt.plot(history.history['val_accuracy'], label='validation', color="blue")
        plt.title('Model accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
    # displaying the model loss
    if disp_mod_loss:
        plt.plot(history.history['loss'], label='train', color="red")
        plt.plot(history.history['val_loss'], label='validation', color="blue")
        plt.title('Model loss')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    score, acc = model.evaluate(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    score, acc = model.evaluate(X_train, y_train)
    print('Train score:', score)
    print('Train accuracy:', acc)

    if predict:
        pred = model.predict(X_test)
        print(pred)
    return model


def train_model_2(X_train, y_train, X_test, y_test, disp_mod_acc=True, disp_mod_loss=True, predict=True):
    # training the model
    model = define_model_2(X_train)
    history = model.fit(
        X_train,
        y_train,
        batch_size=50,
        epochs=200,
        validation_split=0.2,
        shuffle=True
    )
    model.save('C:\Users\Stipe\PycharmProjects\PhotoMathTask\models\model2.h5')
    # Visualising the outcome
    # displaying the model accuracy
    if disp_mod_acc:
        plt.plot(history.history['accuracy'], label='train', color="red")
        plt.plot(history.history['val_accuracy'], label='validation', color="blue")
        plt.title('Model accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
    # displaying the model loss
    if disp_mod_loss:
        plt.plot(history.history['loss'], label='train', color="red")
        plt.plot(history.history['val_loss'], label='validation', color="blue")
        plt.title('Model loss')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    score, acc = model.evaluate(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    score, acc = model.evaluate(X_train, y_train)
    print('Train score:', score)
    print('Train accuracy:', acc)

    if predict:
        pred = model.predict(X_test)
        print(pred)
    return model


def train_model_3(X_train, y_train, X_test, y_test, disp_mod_acc=True, disp_mod_loss=True, predict=True):
    # training the model
    model = define_model_3(X_train)
    history = model.fit(
        X_train,
        y_train,
        batch_size=50,
        epochs=200,
        validation_split=0.2,
        shuffle=True
    )
    model.save('C:\Users\Stipe\PycharmProjects\PhotoMathTask\models\model3.h5')
    # Visualising the outcome
    # displaying the model accuracy
    if disp_mod_acc:
        plt.plot(history.history['accuracy'], label='train', color="red")
        plt.plot(history.history['val_accuracy'], label='validation', color="blue")
        plt.title('Model accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
    # displaying the model loss
    if disp_mod_loss:
        plt.plot(history.history['loss'], label='train', color="red")
        plt.plot(history.history['val_loss'], label='validation', color="blue")
        plt.title('Model loss')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    score, acc = model.evaluate(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    score, acc = model.evaluate(X_train, y_train)
    print('Train score:', score)
    print('Train accuracy:', acc)

    if predict:
        pred = model.predict(X_test)
        print(pred)
    return model



# idx = random.randint(0, len(x_test))
# img = X_test[idx]
# plt.imshow(img.squeeze())
# pred = model.predict(np.expand_dims(img, axis=0))[0]
# ind = (-pred).argsort()[:5]
# latex = [class_names[x] for x in ind]
# print(latex)