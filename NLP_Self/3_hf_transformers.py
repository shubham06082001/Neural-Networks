# Level 1 -> Text Processing
# Level 2 -> Classical NLP
# Level 3 -> Machine Learning for Text
# Level 4 -> Word Embeddings
# Level 5 -> RNN / LSTM
# Level 6 -> Transformers
# Level 7 -> Hugging Face
# Level 8 -> Fine-Tuning
# Level 9 -> Production NLP Project

# Tokenization
sentence = "I love natural language processing NLP!"
tokens = sentence.split()
# for word in set(tokens):
#     print(word, tokens.count(word))

# Remove punctuation
import re
text = "Hello!!! NLP??"
clean = re.sub(r"[^a-zA-Z ]", "", text)
print("Text Cleaning: ",clean)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from gensim.parsing.preprocessing import STOPWORDS

# print(STOPWORDS)

# print(ENGLISH_STOP_WORDS)

words = ["this", "is", "a", "sample", "sentence"]

filtered = [word for word in words if word not in ENGLISH_STOP_WORDS]
print("Filtered Words: ",filtered)
st_words = [word for word in words if word in STOPWORDS]
print("Stopwords: ", st_words)

# Stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

words = ["running", "studies", "hugging", "huggingface"]

for word in words:
    stemmed = stemmer.stem(word)
    print(f"Original: {word}, Stemmed: {stemmed}")

# Lemmatization
import stanza

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma')
doc = nlp("running studies hugging huggingface")
lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
print("Lemmas: ", lemmas)

# Classical NLP Pipeline
# Text -> Cleaning -> Tokenization -> Stopword Removal -> Stemming / Lemmatization -> Feature Extraction (TF-IDF, Word2Vec, GloVe) -> Model Training (SVM, Random Forest, etc.) -> Evaluation

# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I love python",
    "I love NLP"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences)
print("Count Vec - Feature Names: ", vectorizer.get_feature_names_out())
print("Count Vec - Transformed Data: ", X.toarray())

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
print("TF-IDF Feature Names: ", vectorizer.get_feature_names_out())
print("TF-IDF Transformed Data: \n", X.toarray())

# Word Embeddings
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I love learning NLP")
print("Sentiment Analysis Result: ", result)

# Tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "I love natural language processing NLP!"
tokens = tokenizer.tokenize(text)
print("Tokenized Text: ", tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs: ", ids)


# Complete flow
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
inputs = tokenizer("I love natural language processing NLP!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
print("Outputs: ", outputs)
