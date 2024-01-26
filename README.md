# Wrap gensim.doc2vec model

This project help wrap gensim.doc2vec model to export API for my course-mapping project.
APIs help to compare and caculate similarity between two or more documents.

## Install and Test

### 1. Python

[Install](https://www.python.org/downloads/release/python-3121/) python version 3.12.1

### 2. Lib

See in requirements.txt in project and run:

``` cmd
pip install -r requirements.txt
```

### 3. Data for training

[Download](https://drive.google.com/drive/folders/1KvcRU8MzXtL9dXyPkR_kVgRggl49u6h1?usp=sharing) train corpus from my drive. Then put these folders to project like introduce_data folder.
Remember change value of  ```train_corpus_dir``` variable to match with name of directory.

### 4. Run

``` cmd
flask --app app run
```

and go to see <http://127.0.0.1:5000>

### 5. Test API

 See detail in file json in ```test```  folder

## Reference

1. [Gensim Doc2vec model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
2. [Doc2vec paper](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
