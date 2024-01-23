import model
from flask import Flask, request, jsonify
from pyvi import ViTokenizer

model.token_word()
model.train_model()
doc2vec_model = model.load_model()

app = Flask(__name__)

@app.route('/api/compare_all', methods=['POST'])
def api_compare_all():
	data = request.get_json()

	tokenize_data = ViTokenizer.tokenize(data['doc'])
	vec = doc2vec_model.infer_vector(tokenize_data.split(" "))
	res = doc2vec_model.docvecs.most_similar([vec], topn= 2)

	return jsonify(list=res)