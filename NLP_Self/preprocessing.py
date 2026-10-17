"""
NLP PREPROCESSING TUTORIAL (BEGINNER -> ADVANCED)
=================================================

This file demonstrates the most common NLP preprocessing stages.

LIBRARIES USED
--------------
1. re
   - Regular expressions
   - Cleaning text, replacing patterns, removing symbols

2. nltk
   - Most popular educational NLP library
   - Tokenization
   - Stopword removal
   - POS tagging
   - Lemmatization
   - Stemming

3. tokenizers (HuggingFace)
   - Industrial-grade tokenizers
   - BPE (used by GPT-2, RoBERTa, LLaMA variants)
   - WordPiece (used by BERT)

4. requests (optional)
   - Download text from websites

NOTE:
------
Scikit-Learn (sklearn) CAN help with:
  * Text vectorization
      - CountVectorizer
      - TfidfVectorizer
  * Stopword lists

Scikit-Learn CANNOT fully do:
  * POS Tagging
  * Lemmatization
  * Stemming
  * BPE Training
  * WordPiece Training

For complete NLP preprocessing, NLTK / spaCy / HuggingFace Tokenizers
are usually required.
"""

import re
import nltk

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

# Downloads (run once)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")

#####################################################################
# SAMPLE TEXT
#####################################################################

text = """
I'm learning NLP. I don't know if u like it lol!
Running studies are interesting. Apple is GREAT!!!
"""

print("\nORIGINAL TEXT")
print(text)

#####################################################################
# STAGE 1 - LOWERCASE
# Apple -> apple
#####################################################################

text_lower = text.lower()

print("\nSTAGE 1 - LOWERCASE")
print("Apple ->", "Apple".lower())
print(text_lower)

#####################################################################
# STAGE 2 - CONTRACTION EXPANSION
# don't -> do not
#####################################################################

contractions = {
    "i'm": "i am",
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not"
}


def expand_contractions(text):
    for short, full in contractions.items():
        text = re.sub(rf"\b{short}\b", full, text)
    return text

text_contractions = expand_contractions(text_lower)

print("\nSTAGE 2 - CONTRACTION EXPANSION")
print("don't -> do not")
print(text_contractions)

#####################################################################
# STAGE 3 - SLANG EXPANSION
# idk -> i do not know
#####################################################################

slang_map = {
    "idk": "i do not know",
    "u": "you",
    "ur": "your",
    "lol": "laughing out loud"
}


def replace_slang(text):
    for slang, full in slang_map.items():
        text = re.sub(rf"\b{slang}\b", full, text)
    return text

text_slang = replace_slang(text_contractions)

print("\nSTAGE 3 - SLANG EXPANSION")
print("idk -> i do not know")
print(text_slang)

#####################################################################
# STAGE 4 - CLEANING
# hello!!! -> hello
#####################################################################

clean_text = re.sub(r"[^a-z\s]", " ", text_slang)
clean_text = re.sub(r"\s+", " ", clean_text).strip()

print("\nSTAGE 4 - CLEANING")
print("hello!!! -> hello")
print(clean_text)

#####################################################################
# STAGE 5 - TOKENIZATION
# sentence -> words
#####################################################################

# Simplest tokenizer

tokens = clean_text.split()

print("\nSTAGE 5 - TOKENIZATION")
print(tokens)

#####################################################################
# STAGE 6 - STOPWORD REMOVAL
# remove words such as the, is, am, are
#####################################################################

# stop_words = set(stopwords.words("english"))

# tokens = [
#     token
#     for token in tokens
#     if token not in stop_words
# ]

# print("\nSTAGE 6 - STOPWORD REMOVAL")
# print(tokens)

#####################################################################
# STAGE 7 - POS TAGGING
# run(VB), dog(NN)
#####################################################################

pos_tags = pos_tag(tokens)

print("\nSTAGE 7 - POS TAGGING")
print(pos_tags)

#####################################################################
# Convert Penn Treebank tags to WordNet tags
# Needed for powerful lemmatization
#####################################################################


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith("J"):
        return wordnet.ADJ

    if treebank_tag.startswith("V"):
        return wordnet.VERB

    if treebank_tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


#####################################################################
# STAGE 8 - LEMMATIZATION
# running -> run
#####################################################################

lemmatizer = WordNetLemmatizer()

lemmatized_tokens = [
    lemmatizer.lemmatize(word, get_wordnet_pos(tag))
    for word, tag in pos_tags
]

print("\nSTAGE 8 - LEMMATIZATION")
print(lemmatized_tokens)

#####################################################################
# STAGE 9 - STEMMING
# studies -> studi
#####################################################################

porter = PorterStemmer()
snowball = SnowballStemmer("english")

porter_tokens = [porter.stem(word) for word in tokens]
snowball_tokens = [snowball.stem(word) for word in tokens]

print("\nSTAGE 9 - PORTER STEMMING")
print(porter_tokens)

print("\nSTAGE 9 - SNOWBALL STEMMING")
print(snowball_tokens)

#####################################################################
# STAGE 10 - BPE TOKENIZATION
# unbelievable -> un + bel + iev + able
#####################################################################

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=1000,
        special_tokens=["[UNK]"]
    )

    tokenizer.train(["sample_text.txt"], trainer)

    encoded = tokenizer.encode("unbelievable")

    print("\nSTAGE 10 - BPE")
    print(encoded.tokens)

except Exception as ex:
    print("\nBPE section skipped:", ex)

#####################################################################
# STAGE 11 - WORDPIECE TOKENIZATION
# play + ##ing
#####################################################################

try:
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer

    wp_tokenizer = Tokenizer(
        WordPiece(unk_token="[UNK]")
    )

    wp_tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=1000,
        special_tokens=["[UNK]"]
    )

    wp_tokenizer.train([
        "sample_text.txt"
    ], trainer)

    encoded = wp_tokenizer.encode("playing")

    print("\nSTAGE 11 - WORDPIECE")
    print(encoded.tokens)

except Exception as ex:
    print("\nWordPiece section skipped:", ex)

#####################################################################
# SUMMARY
#####################################################################

print("""

PIPELINE SUMMARY
================
1. Lowercase
2. Expand Contractions
3. Expand Slang
4. Clean Text
5. Tokenize
6. Remove Stopwords
7. POS Tagging
8. Lemmatization
9. Stemming
10. BPE
11. WordPiece

Traditional NLP:
    Lowercase -> Tokenization -> Stopwords -> Lemmatization

Modern LLMs:
    Cleaning -> BPE / WordPiece -> Embeddings
""")
