# Neural-POS-Tagger
Neural POS Tagger implemented using PyTorch trained on the Universal Dependencies dataset. The training part has been commented out so that running time of the program will not take long.  
<br/>
The dataset used is : ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-{train,dev,test}.conllu.  
<br/>
We use the first, second and fourth columns only (word index, lowercase word, and
POS tag). The UD dataset does not include punctuation. Thus we filter the input sentence to remove punctuation before tagging it.  
<br/>
The same code present in pos_tagger.py is also present in jupyter notebook format in the file code2.ipynb for each access.

REPORT.md contains the result, scores and their analysis. 

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

# References:
1. https://medium.com/@pankajchandravanshi/nlp-unlocked-pos-tagging-004-447884b6030a
2. https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
3. https://www.kaggle.com/code/krishanudb/lstm-character-word-pos-tag-model-pytorch
4. http://jkk.name/neural-tagger-tutorial/
5. https://github.com/IrwinChay/FNLP-Coursework-And-Note
