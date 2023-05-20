# Scores
Epoch: 01
	Train Loss: 1.978 | Train Acc: 38.46%
	Val. Loss: 152.533 | Val. Acc: 8050.12%
Epoch: 02
	Train Loss: 0.950 | Train Acc: 70.61%
	Val. Loss: 103.582 | Val. Acc: 9248.41%
Epoch: 03
	Train Loss: 0.688 | Train Acc: 78.57%
	Val. Loss: 86.079 | Val. Acc: 10054.91%
Epoch: 04
	Train Loss: 0.574 | Train Acc: 81.99%
	Val. Loss: 77.858 | Val. Acc: 10124.88%
Epoch: 05
	Train Loss: 0.510 | Train Acc: 83.86%
	Val. Loss: 74.251 | Val. Acc: 10187.02%
Epoch: 06
	Train Loss: 0.470 | Train Acc: 84.92%
	Val. Loss: 71.993 | Val. Acc: 10295.89%
Epoch: 07
	Train Loss: 0.441 | Train Acc: 85.77%
	Val. Loss: 70.156 | Val. Acc: 10340.41%
Epoch: 08
	Train Loss: 0.423 | Train Acc: 86.30%
	Val. Loss: 69.292 | Val. Acc: 10452.79%
Epoch: 09
	Train Loss: 0.405 | Train Acc: 86.94%
	Val. Loss: 68.114 | Val. Acc: 10396.52%
Epoch: 10
	Train Loss: 0.391 | Train Acc: 87.30%
	Val. Loss: 67.827 | Val. Acc: 10400.80%

Test Loss: 67.016 |  Test Accuracy: 10126.26%

# Hyperparameter for training model
1. embedding_dim: 100 word embedding dimensions (GLoVe)
2. hidden_dim: 128 features in LSTM hidden state
3. num_layers: 2 layers in LSTM
4. droput: probability of dropping neuron in a connected layer
5. batch size:  128 sentences in each training batch

# Analysis
The scores show performance of the POS tagger over 10 Epochs. 
1. Training loss decreases as the number of epochs increases, indicating that the model is learning and improving its predictions.
2. Training accuracy increases as the number of epochs increases. This indicates that the model is getting better at correctly predicting POS tags.
3. Validation loss also decreases initially, but then starts to level off or even increase.
4. This may indicate that maybe the model is overfitting to the training data.
5. Validation accuracy increases initially, but then starts to level off. This perhaps indicates that the model generalizes well to unseen data, but is not increasingly improving the accuracy.

# classification_report function from sklearn.metrics
{'<pad>': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 4938}, 'NOUN': {'precision': 0.5814134495641345, 'recall': 0.8899213724088635, 'f1-score': 0.7033236041803974, 'support': 4197}, 'PUNCT': {'precision': 0.9931707317073171, 'recall': 0.990593577684074, 'f1-score': 0.991880480675544, 'support': 3083}, 'VERB': {'precision': 0.7133333333333334, 'recall': 0.8129522431259045, 'f1-score': 0.7598917822117011, 'support': 2764}, 'PRON': {'precision': 0.6624843161856964, 'recall': 0.9522091974752029, 'f1-score': 0.781354051054384, 'support': 2218}, 'ADP': {'precision': 0.8756428237494156, 'recall': 0.925852694018784, 'f1-score': 0.9000480538202789, 'support': 2023}, 'DET': {'precision': 0.956386292834891, 'recall': 0.9720316622691293, 'f1-score': 0.9641455116461659, 'support': 1895}, 'PROPN': {'precision': 0.38359543632439097, 'recall': 0.6624068157614483, 'f1-score': 0.48584260886545594, 'support': 1878}, 'ADJ': {'precision': 0.765695067264574, 'recall': 0.7635550586920067, 'f1-score': 0.7646235656311222, 'support': 1789}, 'AUX': {'precision': 0.89875, 'recall': 0.9529489728296885, 'f1-score': 0.9250562881955613, 'support': 1509}, 'ADV': {'precision': 0.882940108892922, 'recall': 0.768562401263823, 'f1-score': 0.8217905405405405, 'support': 1266}, 'CCONJ': {'precision': 0.9909443725743855, 'recall': 0.982051282051282, 'f1-score': 0.9864777849323889, 'support': 780}, 'PART': {'precision': 0.8772727272727273, 'recall': 0.919047619047619, 'f1-score': 0.8976744186046511, 'support': 630}, 'NUM': {'precision': 0.6201923076923077, 'recall': 0.6825396825396826, 'f1-score': 0.6498740554156172, 'support': 378}, 'SCONJ': {'precision': 0.7463556851311953, 'recall': 0.6368159203980099, 'f1-score': 0.687248322147651, 'support': 402}, 'X': {'precision': 0.390625, 'recall': 0.16233766233766234, 'f1-score': 0.2293577981651376, 'support': 154}, 'INTJ': {'precision': 0.37662337662337664, 'recall': 0.5043478260869565, 'f1-score': 0.4312267657992565, 'support': 115}, 'SYM': {'precision': 0.7333333333333333, 'recall': 0.4925373134328358, 'f1-score': 0.5892857142857143, 'support': 67}, 'accuracy': 0.7265505550754504, 'macro avg': {'precision': 0.6915976868046667, 'recall': 0.7261506278568317, 'f1-score': 0.6982834081206426, 'support': 30086}, 'weighted avg': {'precision': 0.6349758235874491, 'recall': 0.7265505550754504, 'f1-score': 0.6710424335312928, 'support': 30086}}