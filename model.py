from os import listdir
from os.path import join
import gensim
from pyvi import ViTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from deep_translator import GoogleTranslator
from langdetect import detect

data = []

stops = open("stopwords_vietnamese.txt", "r", encoding="utf8").readline().split()


def tokenize_word(doc):
    target_doc = doc
    lang = detect(doc)
    if lang != 'vi':
        if len(doc) > 4999:
            translated = ""
            for i in range(0, len(doc), 4999):
                translated += GoogleTranslator(source='auto', target='vi').translate(doc[i:i + 4999])

        else:
            translated = GoogleTranslator(source='auto', target='vi').translate(doc)

            target_doc = translated
    else:
        target_doc = doc

    doc_remove_digit = (''.join([c for c in target_doc if not c.isdigit()])).lower()
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
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    similar_doc = model.docvecs.most_similar(doc_labels[0], )
    model.save("doc2vec.model")


def load_model():
    try:
        return Doc2Vec.load("doc2vec.model")
    except Exception as e:
        print(e)
        return None


def load_data():
    train_corpus_dir = "introduce_data"
    doc_labels = [file for file in listdir(train_corpus_dir) if file.endswith(".txt")]
    # print(doc_labels)
    doc_list = []
    for doc in doc_labels:
        file = open(join(train_corpus_dir, doc), "r", encoding="utf8")
        doc_list.append(file.read())

    return doc_labels, doc_list


def process_data():
    model = load_model()
    doc_labels, doc_list = load_data()

    document_vectors = []
    for i in range(len(doc_list)):
        doc = doc_list[i]
        tokenized_doc = tokenize_word(doc)
        document_vector = model.infer_vector(tokenized_doc)
        document_vectors.append(document_vector)

    return document_vectors
