from os import listdir
from os.path import join
import gensim
from pyvi import ViTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string

data =[]



def token_word():
    stop = open("stopwords_vietnamese.txt", "r", encoding="utf8").read().split()
    for i in range(len(data)):
       data[i] = (''.join([c for c in data[i] if not c.isdigit()])).lower()
       token_words= ViTokenizer.tokenize(data[i]).split()
       token_words_no_punct = [t for t in token_words if t not in stop and t not in string.punctuation]
       data[i] = token_words_no_punct
        

    # print(data)


def train_model():
    train_corpus_dir = "introduce_data"
    docLabels = [file for file in listdir(train_corpus_dir) if file.endswith(".txt")]
    # print(docLabels)

    for doc in docLabels:
        file = open(join(train_corpus_dir,doc), "r", encoding="utf8")
        data.append(file.read())

    token_word()

   # tag data with tags is docLabels
    tagged_data = [TaggedDocument(words=data[i], tags=[docLabels[i]]) for i in range(len(data))]
    print(tagged_data)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, dm=1, window=5, min_count=2, epochs=100)

    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    similar_doc = model.docvecs.most_similar(docLabels[0])
    print(similar_doc)
    model.save("doc2vec.model")


def load_model():
    try:
        return Doc2Vec.load("doc2vec.model")
    except:
        print("model is not found")
        return None
