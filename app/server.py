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
from nltk.corpus import stopwords
import string
from random import randrange

#export_file_url = 'https://www.googleapis.com/drive/v3/files/1-hcOTjAD1ELR6_1FPhW7arifG9jg1Q8N?alt=media&key=AIzaSyDMZOVdakuqXD_IBalpDK43XVTQAA8Ja2Q'
#export_file_url = 'https://www.googleapis.com/drive/v3/files/1-02L2PRi2fnE7QTccwpH2A3s_vA39wZe?alt=media&key=AIzaSyCIaEnZ46EdMleKdmKeBRZNFpd_yRTQiuU'
export_file_url = 'https://www.googleapis.com/drive/v3/files/1-RcEJYzZ2g3k51QLZZ-6RzDXbdJL6BRJ?alt=media&key=AIzaSyCiLpxA2j60lVqzx-kehWiISSn_Lsgk0CE'
export_file_name = 'epochs:2-lstm_dim:100-lstm_layers:1-devacc:0.444.pth'

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

'''
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()
'''
parser = argparse.ArgumentParser()
parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
parser.add_argument("--EMBEDDING_DIM", "-ed", default=300, type=int)
parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_false")
parser.add_argument("--AUXILIARY_TASK", "-aux", default="negation_scope")
parser.add_argument("--EMBEDDINGS", "-emb",
        default="../../project/embeddings/embeddings/google.txt")
parser.add_argument("--DATA_DIR", "-dd",
        default="../data/datasets/en")
parser.add_argument("--DATASET", "-data",
        default="SST")
parser.add_argument("--AUXILIARY_DATASET", "-auxdata",
        default="preprocessed/starsem_negation/cdt.conllu")
parser.add_argument("--SENTIMENT_LR", "-slr", default=0.001, type=float)
parser.add_argument("--AUXILIARY_LR", "-alr", default=0.0001, type=float)
parser.add_argument("--FINE_GRAINED", "-fg",
        default="fine",
        help="Either 'fine' or 'binary' (defaults to 'fine'.")

#args = parser.parse_args()
#print(args)
args, unknown = parser.parse_known_args()

START_TAG = "<START>"
STOP_TAG = "<STOP>"


# Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
embeddings = WordVecs(args.EMBEDDINGS)
print("loaded embeddings from {0}".format(args.EMBEDDINGS))
w2idx = embeddings._w2idx

# Create shared vocabulary for tasks
vocab = Vocab(train=True)

# Update with word2idx from pretrained embeddings so we don't lose them
# making sure to change them by one to avoid overwriting the UNK token
# at index 0
with_unk = {}
for word, idx in embeddings._w2idx.items():
    with_unk[word] = idx + 1
vocab.update(with_unk)

# Import datasets
# This will update vocab with words not found in embeddings
datadir = os.path.join(args.DATA_DIR, args.DATASET, args.FINE_GRAINED)
sst = SSTDataset(vocab, False, datadir)

maintask_train_iter = sst.get_split("train")
maintask_dev_iter = sst.get_split("dev")
maintask_test_iter = sst.get_split("test")

maintask_loader = DataLoader(maintask_train_iter,
                             batch_size=args.BATCH_SIZE,
                             collate_fn=maintask_train_iter.collate_fn,
                             shuffle=True)

if args.AUXILIARY_TASK in ["speculation_scope"]:
    X, Y, org_X, org_Y, word2id, char2id, task2label2id =\
     get_conll_data(os.path.join(args.DATA_DIR, "preprocessed/SFU/filtered_speculation_scope.conll"),
                    ["speculation_scope"],
                    word2id=vocab)


if args.AUXILIARY_TASK in ["negation_scope"]:
    X, Y, org_X, org_Y, word2id, char2id, task2label2id =\
     get_conll_data(os.path.join(args.DATA_DIR, args.AUXILIARY_DATASET),
                    ["negation_scope"],
                    word2id=vocab)


if args.AUXILIARY_TASK in ["xpos", "upos", "multiword", "supersense"]:
    X, Y, org_X, org_Y, word2id, char2id, task2label2id =\
    get_conll_data(os.path.join(args.DATA_DIR, "preprocessed/streusle/train/streusle.ud_train.conllulex"),
                   ["xpos", "upos", "multiword", "supersense"],
                   word2id=vocab)


if args.AUXILIARY_TASK not in ["None", "none", 0, None]:
    train_n = int(len(X) * .9)
    tag_to_ix = task2label2id[args.AUXILIARY_TASK]
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)

    X, char_X = zip(*X)

else:
    # Set all relevant auxiliary task parameters to None
    tag_to_ix = {"None": 0}
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)
    task2label2id = None

    auxiliary_trainX = None
    auxiliary_trainY = None
    auxiliary_testX = None
    auxiliary_testY = None


# Get new embedding matrix so that words not included in pretrained embeddings have a random embedding

diff = len(vocab) - embeddings.vocab_length - 1
UNK_embedding = np.zeros((1, 300))
new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))

model = Hierarchical_Model(vocab,
                           new_matrix,
                           tag_to_ix,
                           len(sst.labels),
                           task2label2id,
                           300,
                           100,
                           1,
                           train_embeddings="-te")

#model.load_state_dict(torch.load(path / export_file_name))
model = torch.load(path / export_file_name)


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

def predict_sentiment(review, vocab, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encoded
    #encoded = tokenizer.texts_to_matrix([line], mode='freq')
    encoded = torch.LongTensor(vocab.ws2ids(review))
    #encoded = Split([torch.LongTensor(vocab.ws2ids(review)), torch.LongTensor([int(1)])])
    yhat = 1
    try:
        for sents, targets in DataLoader (encoded, 1,
                                         #collate_fn=encoded.collate_fn,
                                         shuffle=False):
            yhat = model.predict_sentiment(sents)
    except Exception as e:
        yhat = randrange(4)
    print(yhat)
    return round(yhat)

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    print('inside analyze')
    text_data = await request.form()
    print(text_data)
    text_bytes = text_data['file']
    #img = open_image(BytesIO(img_bytes))
    data = torch.LongTensor(vocab.ws2ids(text_bytes))
    single_iter = sst.get_split("test")
    model.eval()
    preds = predict_sentiment(text_bytes, vocab, model)
    print(preds)
    return JSONResponse({'result': str(preds)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
