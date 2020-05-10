# Machine Translation
Python Flask web app which hosts a deep-learning Keras neural network to translate simple phrases in French, Italian, Spanish, German and Turkish into English.  This project serves as my final independent student project for the Flatiron School Data Science curriculum.  

#### -- Project Status: [Inactive]

## Project Intro/Objective
This project had four main goals which are complete:
* Can I build a machine-translation model at all in Python?
* How successful are the translations?
* Can I implement AWS to build it?
* Can I host interactive translations on a local website in Flask?
### Next Steps (as of August 2019)
* Can I improve the accuracy with other models like gensim’s Word2Vec? And Spacy's POS, entities, etc.?
* Can I port the web app from my local host to an AWS instance 
* Can I port the project to SageMaker on AWS?
* Can I refactor the code easilt to reverse the translation direction _from English_ to the target language?

### Methods Used
* Neural Networks

### Technologies
##### for machine translation
* Python
* Keras (layers: LSTM, Embedding, RepeatVector and TimeDistributed)
* nltk.bleu_scores
* AWS EC2 cloud computing instance TensorFlow(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN): source activate tensorflow_p36
* AWS S3 Bucket storage
#### for web app 
* Python
* Flask
* HTML
* JavaScript
* CSS

## Project Description
### Neural Net
This project investigates how NLP _encoding techniques_ can help improve machine translations.  In order to start that investigation, a basic machine translation model must be built as a first step.  In this case, I used the following Keras layers: LSTM, Embedding, RepeatVector and TimeDistributed.  

### NLP Encoding
Also for the initial baseline, I implemented the most simple text-cleaning and encoding methods:  
* all lower-case, no punctutation, no funny characters, etc.  
* keras.tokenizer.texts_to_sequences()

The end-goal however, is to make this encoding step much more sophisticated in terms of contemporary technologies. Can sophisticated encodings improved translation accuracy?
* Gensim's Word2Vec
* Spacy's POS, entities, etc.

### Data Sources
The training data for this project came from extremely clean and pre-translated sentence pairs published by the Tatoeba Project
* http://www.manythings.org/anki/ 
* I limited the number of translated phrases to 75,000 for each language pair.  

## Needs of this project

- Python NLP specialist: word embeddings, Word2Vec Doc2Vec Seq2Seq, etc.
- Python Deep-Learning specialist:  keras, AWS (EC2, SageMaker)
- Python frontend developers (flask, HTML, CSS, JS)

## Getting Started

1. TK

## Featured Notebooks/Analysis/Deliverables
* TK
