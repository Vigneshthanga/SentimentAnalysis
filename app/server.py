import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
#from fastai.data.all import *
#from fastai.optimizer import *
#from fastai.callback.core import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from hierarchical_model import *
import pathlib

#export_file_url = 'https://www.googleapis.com/drive/v3/files/1-hcOTjAD1ELR6_1FPhW7arifG9jg1Q8N?alt=media&key=AIzaSyDMZOVdakuqXD_IBalpDK43XVTQAA8Ja2Q'
export_file_url = 'https://www.googleapis.com/drive/v3/files/1-02L2PRi2fnE7QTccwpH2A3s_vA39wZe?alt=media&key=AIzaSyCIaEnZ46EdMleKdmKeBRZNFpd_yRTQiuU'
export_file_name = 'model.pt'

classes = ['1', '2', '3', '4']
path = pathlib.Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        model = Hierarchical_Model(vocab,
                                   new_matrix,
                                   tag_to_ix,
                                   num_labels,
                                   task2label2id,
                                   embedding_dim,
                                   hidden_dim,
                                   1,
                                   train_embeddings=train_embeddings)
#        learn = learner.load(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['text1'].read())
    #img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img_bytes)
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
