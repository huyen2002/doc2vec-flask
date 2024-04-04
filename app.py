from deep_translator import GoogleTranslator
from pyvi.ViTokenizer import ViTokenizer

import model
from flask import Flask, request, jsonify
import gensim

doc2vec_model = model.load_model()
print("hello")
app = Flask(__name__)

print(ViTokenizer.tokenize("Bạn dạo này có khỏe không?").split())

doc_labels, doc_list = model.load_data()
other_document_vectors = model.process_data()


# @app.route('/api/top_n_most_similar', methods=['POST'])
# def api_get_top_n_similar():
#     if request.is_json:
#         data = request.get_json()
#         # doc2vec_model.random.seed(0)
#         vec = doc2vec_model.infer_vector(model.tokenize_word(data['data']))
#         print(vec)
#         res = doc2vec_model.docvecs.most_similar([vec], topn=8)
#         return jsonify(list=res)
#     else:
#         return jsonify(error="Request is not json"), 400


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


@app.route('/api/top_n_most_similar', methods=['POST'])
def api_get_top_8_similar():
    similarities = []
    if request.is_json:
        data = request.get_json()
        doc2vec_model.random.seed(0)
        target_document_vector = doc2vec_model.infer_vector(model.tokenize_word(data['data']))
        vec_target = gensim.matutils.full2sparse(target_document_vector)
        for other_vector in other_document_vectors:
            vec_other = gensim.matutils.full2sparse(other_vector)
            similarity = gensim.matutils.cossim(vec_target, vec_other)
            similarities.append(similarity)
        # Find the indices of the top 5 most similar documents
        top_5_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:8]
        # Get the top 5 most similar documents label and similarity
        top_5_similar_documents = [(doc_labels[i], similarities[i]) for i in top_5_indices]

        return jsonify(list=top_5_similar_documents)
    else:
        return jsonify(error="Request is not json"), 400
