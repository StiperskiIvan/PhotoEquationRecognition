# PhotoEquationRecognition
Using tensorflow to classify and calculate a simple mathematical equation from input image

Whole solution is contained in the Photo folder in app.py
you can run it in one of the IDEs avaliable, like Pycharm, VisualStudio etc.

In folder tools there are additional scripts for training different network architectures for object detection as well as app.py script without the flask web part so that you can test it on a local image or retrain the model, you just need to adjust the paths to images, training data and model

To start the app do the following:

```sh
docker build  https://github.com/StiperskiIvan/PhotoEquationRecognition.git#master -t photo
```
```sh
docker container create --name photo_container --publish 5000:5000 photo
```
```sh
docker container start photo_container
```

If for some reason Docker Image does not work, dowload or clone the code from github and run Photo/app.py in your IDE
The link for a localhost:5000 port will apear and you can just click on it and copy it in your favorite browser

Tap on the button file and chose an image to upload and click submit

If the solution is not correct please be reasonable, the network is not perfect :)

tensorflow CNN model contained 17 training labels: ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'times'] preprocessed and isolated from kagle handwritten-math symbol and digit dataset https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset

To have a a trained model in a reasonable time the amount of data was minimised so that all individual classes have thee same amount of data as the class with minimal number od image data which resulted in 878 images per class which was split 80:20 to training and validation data.

The most important metric was test accuracy, which was around 80%, the results also showed that some classes like div, 2, -, 7 had the score of about 70% accuracy 

Input image was a tensor of size (1, 100, 100, 3)

![Image of a network](/Photo/model.png)
