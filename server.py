from fastapi import FastAPI
from uvicorn import run

from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims

from model import FaceModel


app = FastAPI()

INPUT_SHAPE = (1, 198, 198, 3)
model_dir = './my_checkpoint.h5'
model = FaceModel()
model.build(input_shape=INPUT_SHAPE)
model.load_weights(model_dir)


@app.get("/")
async def root():
    return {"message": "Welcome to the Face Age API!"}


@app.post('/predict')
async def predict(image_link: str = ''):
    if image_link == '':
        return {'message': 'No image link provided'}

    img_path = get_file(origin=image_link)
    img = load_img(img_path, target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    pred = int(model.predict(img_array / 255))

    return {'model-prediction-age': pred}


if __name__ == '__main__':
    # run server using given host and port
    run(app, host='127.0.0.1', port=80)

