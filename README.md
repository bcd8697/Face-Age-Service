<h1 align="center"> Face Age Service :camera: :boy: :underage: </h1>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#setup">Setup</a> •
  <a href="#content">Content</a> 
</p>

## Description

An example of the implementation of a face detector with age detection. The model for deployment is assembled into a dockerized service on FastAPI with Swagger, which accepts POST requests in the form of hyperlinks and returns JSON with the results of the model. <a href="https://susanqq.github.io/UTKFace/" target="_blank"> UTKFace dataset </a> was used for training.

## Dependencies

* Python 3
* <a href="https://hub.packtpub.com/python-data-stack/" target="_blank"> Python data stack </a>
* <a href="https://www.tensorflow.org/" target="_target"> Tensorflow 2.0 </a>
* <a href="https://numpy.org/" target="_target"> NumPy </a>
* <a href="https://github.com/tqdm/tqdm" target="_target"> tqdm </a>
* <a href="https://fastapi.tiangolo.com/" target="_target"> FastAPI </a> 
* <a href="https://www.uvicorn.org/" target="_target"> Uvicorn </a>
* <a href="https://www.docker.com/" target="_target"> Docker </a> 

## Setup

Clone GitHub repository:

```
git clone https://github.com/bcd8697/FaceAgeService
```

Make sure that all dependencies are installed and run the respective notebooks, containing the experiments for each model.
For easy installation you may try to use:

```
pip install -r requirements.txt
```

After that you will need to build docker image and to run docker container locally.

For building image move to the "FaceAgeService" directory via the CLI, choose name for your server and do:

```
docker build . -t <app_server_name>
```

After that the building process will run. It may take a few minutes, so, please, be patient.

When image is ready, you need to run it as a docker container. For this you may do:

```
docker run --rm -it -p <your_port>:80 <app_server_name>
```

In this application, to run, you must select a free port on the local machine. It is suggested to use port 80 as default. To do this, run the command above, replacing <your_port> with 80 (without quotes). Port 80 is also chosen as a target by default.

When app is ready and docker container is running, you may try to open your web-browser and type:

```
https://localhost:80
```

This should lead you to the text page with "Welcome to Face Age API", if everything is done correctly.

After that you may want to try the app, using Swagger. For this simply open web-page:

```
https://localhost:80/docs
```

You need to open POST method and try to use it. Copying web-link to the textbox, you may see that the app returns a JSON below which contains detected age of a person on the picture.

Please, pay attention, that this app is done only to demonstrate basic priciples of neural network production deployment and does not guarantee high accuracy at all!

## Content

This repository contains the codebase for the project, its structure is the following:

* **Dockerfile** docker image build file
* **analysis_and_training** Jupyter notebook with GPU training process description (done with Google Colab)
* **model.py** python file with model class and helper class
* **my_checkpoint.h5** TF2 checkpoint example
* **requirements.txt** file with all packages needed to build docker image properly
* **server.py** python file with FastAPI application, contains model call and server methods
* **utils.py** python file with helper functions, used for data preparation and training
