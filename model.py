from os import listdir
from os.path import join
import gensim
from pyvi import ViTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from deep_translator import GoogleTranslator

data = []

stops = open("stopwords_vietnamese.txt", "r", encoding="utf8").read().split()


def translate_file():
    translated = GoogleTranslator(source='auto', target='vi').translate_file('introduce_data/UK_BSC_I100.txt')
    print(translated)


def tokenize_word(doc):
    doc_remove_digit = (''.join([c for c in doc if not c.isdigit()])).lower()
    token_words = ViTokenizer.tokenize(doc_remove_digit).split()
    token_words_no_punct = [t for t in token_words if t not in stops and t not in string.punctuation]
    return token_words_no_punct


def train_model():
    train_corpus_dir = "introduce_data"
    doc_labels = [file for file in listdir(train_corpus_dir) if file.endswith(".txt")]
    # print(doc_labels)

    for doc in doc_labels:
        if doc.startswith('VN'):
            file = open(join(train_corpus_dir, doc), "r", encoding="utf8")
            data.append(file.read())
        else:
            translated = GoogleTranslator(source='auto', target='vi').translate_file(join(train_corpus_dir, doc))
            data.append(translated)

    for i in range(len(data)):
        data[i] = tokenize_word(data[i])

        # tag data with tags is doc_labels
    tagged_data = [TaggedDocument(words=data[i], tags=[doc_labels[i]]) for i in range(len(data))]
    # print(tagged_data)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, dm=1, window=5, min_count=2, epochs=100, workers=5)

    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs )

    similar_doc = model.docvecs.most_similar(doc_labels[0])
    print(similar_doc)
    model.save("doc2vec.model")


def load_model():
    try:
        return Doc2Vec.load("doc2vec.model")
    except:
        print("model is not found")
        return None
