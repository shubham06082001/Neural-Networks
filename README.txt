Papers referenced:
Word2Vec: https://arxiv.org/pdf/1301.3781
GRUs: https://arxiv.org/abs/1412.3555
LSTMS: https://www.bioinf.jku.at/publication...
Seq2Seq: https://arxiv.org/abs/1409.3215
Enc-Dec Attention: https://arxiv.org/abs/1409.0473
Attention is All You Need: https://arxiv.org/abs/1706.03762
GPT: https://cdn.openai.com/research-cover...
BERT: https://arxiv.org/pdf/1810.04805.pdf
Relative Position Embeddings: https://arxiv.org/abs/1803.02155
Transformer-XL: https://arxiv.org/abs/1901.02860
T5: https://arxiv.org/abs/1910.10683
XL-Net: https://arxiv.org/abs/1906.08237
Performer: https://arxiv.org/abs/2009.14794
Linformer: https://arxiv.org/abs/2006.04768
Longfromer: https://arxiv.org/abs/2004.05150
LORA: https://arxiv.org/abs/2106.09685
GPT-3: https://arxiv.org/abs/2005.14165
InstructGPT: https://arxiv.org/abs/2203.02155
GPT-4 Technical Report: https://arxiv.org/abs/2303.08774



pip3 install nltk requests
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"

python -c "import requests, os; os.makedirs('datasets', exist_ok=True); url='https://www.gutenberg.org/cache/epub/42324/pg42324.txt'; r=requests.get(url); r.raise_for_status(); open('datasets/frankenstein.txt','w',encoding='utf-8').write(r.text); print('File saved successfully!')"



SMALL < 1GB----

.venv/bin/python -m pip install -r requirements.txt nltk gensim datasets transformers huggingface-hub tensorflow-hub stanza certifi

SSL_CERT_FILE="$(.venv/bin/python -m certifi)" REQUESTS_CA_BUNDLE="$(.venv/bin/python -m certifi)" .venv/bin/python - <<'PY'
import nltk
import stanza
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tensorflow import keras

for resource in [
    "stopwords",
    "punkt",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource)

stanza.download("en")
load_dataset("banking77")

for model in [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "cardiffnlp/twitter-roberta-base-sentiment",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    "distilbert-base-cased-distilled-squad",
    "gpt2",
    "google/vit-base-patch16-224",
    "facebook/detr-resnet-50",
    "facebook/detr-resnet-50-panoptic",
    "microsoft/resnet-50",
]:
    snapshot_download(model)

keras.datasets.mnist.load_data()
keras.datasets.fashion_mnist.load_data()
keras.datasets.boston_housing.load_data()
keras.applications.ResNet50(weights="imagenet")
PY





LARGE > 1GB -----

SSL_CERT_FILE="$(python -m certifi)" REQUESTS_CA_BUNDLE="$(python -m certifi)" python - <<'PY'
import gensim.downloader as gensim_api
from huggingface_hub import snapshot_download
import tensorflow_hub as hub

gensim_api.load("word2vec-google-news-300")
snapshot_download("LiquidAI/LFM2-1.2B")
snapshot_download("bigscience/bloom")
hub.load("https://tfhub.dev/deepmind/biggan-512/1")
PY