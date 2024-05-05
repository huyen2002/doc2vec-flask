import numpy as np
import model
from flask import Flask, request, jsonify
import gensim
import logging

doc2vec_model = model.load_model()
doc2vec_model_dbow = model.load_model_dbow()
print("hello")
app = Flask(__name__)


@app.route('/api/compare_two_documents', methods=['POST'])
def api_compare_two_documents():
    data = request.get_json()
    doc2vec_model.random.seed(0)

    vec_1 = doc2vec_model.infer_vector(model.tokenize_word(data['document_1']))
    doc2vec_model.random.seed(0)
    vec_2 = doc2vec_model.infer_vector(model.tokenize_word(data['document_2']))

    vec_1 = gensim.matutils.full2sparse(vec_1)
    vec_2 = gensim.matutils.full2sparse(vec_2)

    similarity = gensim.matutils.cossim(vec_1, vec_2)
    print(similarity)
    return jsonify(similarity=similarity)


@app.route('/api/infer_vector', methods=['POST'])
def api_infer_vector():
    if request.is_json:
        data = request.get_json()
        # print('hello', data)
        doc2vec_model.random.seed(0)
        print(model.tokenize_word(data['data']))
        vec = doc2vec_model.infer_vector(model.tokenize_word(data['data']))
        # print(vec)
        # convert numpy array to json serializable list
        vec = vec.tolist()
        return jsonify(vector=vec)
    else:
        return jsonify(error="Request is not json"), 400


@app.route('/api/cossim_between_two_vectors', methods=['POST'])
def api_cossim_between_two_vectors():
    if request.is_json:
        data = request.get_json()
        vec_1 = np.array(data['vector1'])
        vec_2 = np.array(data['vector2'])
        vec_1 = gensim.matutils.full2sparse(np.array(vec_1))
        vec_2 = gensim.matutils.full2sparse(np.array(vec_2))
        similarity = gensim.matutils.cossim(vec_1, vec_2)
        return jsonify(similarity)
    else:
        return jsonify(error="Request is not json"), 400


@app.route('/api/infer_vector_dbow', methods=['POST'])
def api_infer_vector_dbow():
    if request.is_json:
        data = request.get_json()
        doc2vec_model_dbow.random.seed(0)
        vec = doc2vec_model_dbow.infer_vector(model.tokenize_word(data['data']))
        vec = vec.tolist()
        return jsonify(vector=vec)
    else:
        return jsonify(error="Request is not json"), 400


@app.route('/api/compare_two_documents_dbow', methods=['POST'])
def api_compare_two_documents_dbow():
    data = request.get_json()
    doc2vec_model_dbow.random.seed(0)

    vec_1 = doc2vec_model_dbow.infer_vector(model.tokenize_word(data['document_1']))
    doc2vec_model_dbow.random.seed(0)
    vec_2 = doc2vec_model_dbow.infer_vector(model.tokenize_word(data['document_2']))

    vec_1 = gensim.matutils.full2sparse(vec_1)
    vec_2 = gensim.matutils.full2sparse(vec_2)

    similarity = gensim.matutils.cossim(vec_1, vec_2)
    print(similarity)
    return jsonify(similarity=similarity)
