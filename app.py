import model
from flask import Flask, request, jsonify
from pyvi import ViTokenizer
import gensim

# model.translate_file()
model.train_model()
doc2vec_model = model.load_model()

app = Flask(__name__)


@app.route('/api/top_n_most_similar', methods=['POST'])
def api_get_top_n_similar():
    if request.is_json:
        data = request.get_json()
        print(data)
        vec = doc2vec_model.infer_vector(model.tokenize_word(data['data']))
        print(vec)
        res = doc2vec_model.docvecs.most_similar([vec], topn=5)
        return jsonify(list=res)
    else:
        return jsonify(error="Request is not json"), 400


@app.route('/api/compare_two_documents', methods=['POST'])
def api_compare_two_documents():
    data = request.get_json()
    vec_1 = doc2vec_model.infer_vector(model.tokenize_word(data['doc_1']))
    vec_2 = doc2vec_model.infer_vector(model.tokenize_word(data['doc_2']))

    vec_1 = gensim.matutils.full2sparse(vec_1)
    vec_2 = gensim.matutils.full2sparse(vec_2)

    similarity = gensim.matutils.cossim(vec_1, vec_2)
    return jsonify(2)
