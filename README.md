# PhotoEquationRecognition
Using tensorflow to classify and calculate a simple mathematical equation from input image

Whole solution is contained in the Photo folder in app.py



docker build  https://github.com/StiperskiIvan/PhotoEquationRecognition#master:Photo -t photo
docker container create --name photo_container --publish 5000:5000 photo
docker container start photo_container
