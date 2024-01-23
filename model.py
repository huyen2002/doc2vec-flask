import gensim
from pyvi import ViTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

document_1 = "Ngành công nghệ thông tin có chương trình học bao gồm 124 tín chỉ. Trong đó 120 tín chỉ là bắt buộc, 4 tín chỉ là tự chọn. Nhằm tạo ra chất lượng đào tạo cao nhất."

document_2 = "Ngành Trí tuệ nhân tạo có chương trình học bao gồm 120 tín chỉ. Ứng dụng AI vào công nghệ thông tin đang là xu thế hiện nay. "

document_3 = "Chương trình Ai là triệu phú được phát sóng trên vtv3."

data = [document_1, document_2, document_3]


def token_word():
    for i in range(len(data)):
        data[i] = ViTokenizer.tokenize(data[i])

    print(data)


def train_model():
    tagged_data = [TaggedDocument(words=data[i].split(), tags=[str(i)]) for i in range(len(data))]
    print(tagged_data)
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, dm=1, window=5, min_count=2, epochs=100)

    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    vec = model.infer_vector(data[0].split(" "))

    similar_doc = model.docvecs.most_similar('0')
    print(similar_doc)
    model.save("doc2vec.model")


def load_model():
    try:
        return Doc2Vec.load("doc2vec.model")
    except:
        print("model is not found")
        return None
