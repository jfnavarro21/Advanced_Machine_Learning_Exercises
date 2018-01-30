
# In[1]:


# Import dependencies 
#%matplotlib inline
import numpy as np
from keras.layers import Dense, Activation, LSTM, Input
from keras.models import Sequential, Model
from keras.optimizers import Adagrad, adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.layers import Embedding
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer 
import pandas as pd
import os
import re




# # Read the data

# In[2]:


# Read in the train data (1000 rows)
dataPath = "./data/"
#train = pd.read_csv(dataPath+'quora_train_1000.csv',usecols=['question1','question2'])
train = pd.read_csv(dataPath+'train_sample.csv',usecols=['question1','question2'])
train.dropna(inplace=True) # remove two rows as in NLP feature creation
train = train[:1000] # only for demo and testing, comment out with complete data
train.head()


# # Steps of Preprocessing
# 
# Description of all Keras tools necessary for converting questions into additional useful features for neural network is over. 
# 
# Now go through the several following steps of processing questions. 
# 
# ## Step 1. Lemmatization
# 
# Questions are preprocessed so that the different forms of writing the same text (like "don't" and "do not") are  matched. Lemmatization similar to one done in the first part of the project helps again. 
# 
# Lemmatize with *WordNetLemmatizer*:

# In[3]:


# Create cutter function
from nltk.stem.wordnet import WordNetLemmatizer
WNL = WordNetLemmatizer()

def cutter(word):
    if len(word) < 4:
        return word
    return WNL.lemmatize(WNL.lemmatize(word, "n"), "v")


# In[4]:


# Create preprocess function (uses cutter)

def preprocess(string):
    # standardize expression with apostrophe, replace some special symbols with word
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")         .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")         .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")         .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")         .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")         .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")         .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    # remove punctuation and special symbols
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    # lemmatize
    string = ' '.join([cutter(w) for w in string.split()])
    return string


# Apply preprocessing to train sample. 
# 
# All transformations applied to train should be applied to test too.

# In[5]:


# run preprocess function on all of train set

print('Question 1: %s' % train["question1"][1])
print('Question 2: %s' % train["question2"][1])
train["question1"] = train["question1"].fillna("").apply(preprocess)
train["question2"] = train["question2"].fillna("").apply(preprocess)
print('Question 1 processed: %s' % train.question1[1])
print('Question 2 processed: %s' % train.question2[1])


# ## Step 2. Creating vocabulary of frequent words
# 
# Create vocabulary of relatively frequent words in questions: words with frequency greater than *MIN_WORD_OCCURRENCE* times. 
# 
# For the small dataset *MIN_WORD_OCCURRENCE* is selected small, but for the whole dataset it should be much larger (may be in the range 50-150).
# 
# For word count use familiar *CountVectorizer*.

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer

MIN_WORD_OCCURRENCE = 100 # 3 for demo and testing in local environment. Select number for final results

all_questions = pd.Series(train["question1"].tolist() + train["question2"].tolist()).unique()
vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", # replace white spaces with spaces
                             min_df=MIN_WORD_OCCURRENCE)
vectorizer.fit(all_questions)
top_words = set(vectorizer.vocabulary_.keys())
print(len(top_words),'top_words')
print('Top words %s' % list(top_words)[:10])


# ## Step 3. Remove rare words
# 
# The consecutive rare words are replaced with one word "suspense" (you may try another replacement). The result is limited to 30 trailing words. 
# 
# Remove first words in long question since the end of it is usually more important. 
# 
# Add "suspense" to *top_words*.

# In[7]:


REPLACE_WORD = "suspense"
top_words.add(REPLACE_WORD)
MAX_SEQUENCE_LENGTH = 30


# In[8]:


def prepare(q):
    new_q = []
    new_suspense = True # ready to add REPLACE_WORD 
    # a[::-1] invert order of list a, so we start from the end
    for w in q.split()[::-1]:
        if w in top_words:
            new_q = [w] + new_q # add word from top_words
            new_suspense = True
        elif new_suspense:
            new_q = [REPLACE_WORD] + new_q
            new_suspense = False  # only 1 REPLACE_WORD for group of rare words
        if len(new_q) == MAX_SEQUENCE_LENGTH:
            break
    new_q = " ".join(new_q)
    return new_q

question = train.question1[9]
print('Question: %s' % question)
print('Prepared question: %s' % prepare(question))


# Apply the function to train questions

# In[9]:


q1s_train = train.question1.apply(prepare)
q2s_train = train.question2.apply(prepare)
print(q1s_train[0])


# ## Step 4. Create embedding index
# 
# Build embedding index - dictionary with words from *top_words* as keys and their vector presentations as values.
# 
# Take vector presentations of words from Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) embedding file [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip). Each line of the file contains word space separated from components of word vector.

# In[10]:


EMBEDDING_DIM = 300
EMBEDDING_FILE = "./glove.840B.300d.txt"

def get_embedding():
    embeddings_index = {}
    with open(EMBEDDING_FILE, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) == EMBEDDING_DIM + 1 and word in top_words:
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
    return embeddings_index


# Build *embeddings_index* and reduce *top_words* to those having vector representation.

# In[11]:


embeddings_index = get_embedding()
print("Words not found in the embedding:", top_words - embeddings_index.keys())
top_words = embeddings_index.keys()


# ## Step 5. Transform questions into integer valued sequences of equal lengths
# 
# It is described above how *Tokenizer.texts_to_sequences* converts question to a list of integers. 
# 
# But such lists may have different lengths for different questions. 
# 
# Keras provides method for fixing this issue:
# 
# *keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.)* 
# 
# It transforms a list of *num_samples* sequences (lists of scalars) into a 2D Numpy array of shape 
# 
# *(num_samples, num_timesteps)*, 
# 
# where *num_timesteps* is either *maxlen* argument (if provided), or the length of the longest sequence.
# 
# Sequences that are shorter than *num_timesteps* are padded with *value* at the end. Sequences longer than *num_timesteps* are truncated so that they have the desired length. 
# 
# Position where padding or truncation happens is determined by *padding* or *truncating*, respectively.
# 
# Here are several examples

# In[12]:


from keras.preprocessing.sequence import pad_sequences
sequences = [[1,2],[1,2,3,4,5]]
print('Original sequences: %s' % sequences)
print('Padded default: %s' % pad_sequences(sequences))
print('Padded with maxlen=4: %s' % pad_sequences(sequences,maxlen=4))
print('Padded with maxlen=4, padding=post: %s' % pad_sequences(sequences,maxlen=4,padding='post'))
print('Padded with maxlen=4, padding=post, truncating=post: %s'       %pad_sequences(sequences,maxlen=4,padding='post',truncating='post'))


# Fit *Tokenizer* to the questions obtained after Step 3 and apply *texts_to_sequences* and *pad_sequences* to them.

# In[13]:


tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(q1s_train, q2s_train))
word_index = tokenizer.word_index
data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), 
                       maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), 
                       maxlen=MAX_SEQUENCE_LENGTH)
print('Final representation of first question 1:')
print(data_1[:3])
print('Final representation of first question 2:')
print(data_2[0])
len(data_1)


# Each question now is represented by a vector of 30 numbers.
# 
# Repeat the same steps with *test* set and create:
# 
# *q1s_test -> test_data_1*  
# *q2s_test -> test_data_2*  
# 
# Do not refit Tokenizer, use the same as for *train*.

# In[14]:


test = pd.read_csv(dataPath+'test_sample.csv',usecols=['question1','question2'])
test["question1"] = test["question1"].fillna("").apply(preprocess)
test["question2"] = test["question2"].fillna("").apply(preprocess)
q1s_test = test.question1.apply(prepare)
q2s_test = test.question2.apply(prepare)
test_data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_test), 
                       maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_test), 
                       maxlen=MAX_SEQUENCE_LENGTH)


# ## Step 6. Create embedding matrix
# 
# Now make embedding matrix of weights from embedding index. 
# 
# The *i-th* row of this matrix is a vector representation of word with index *i* in *word_index*. 
# 
# The embedding matrix will be used as weights matrix for embedding layer.

# In[15]:


nb_words = len(word_index) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))  # matrix of zeros

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Create embedding layer from embedding matrix as follows.

# In[16]:


embedding_layer = Embedding(nb_words, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# Setting *trainable=False* declares that no changing weights is required during traning. 
# 
# This layer just transforms sequences of integers (word indexes) into sequences of their vector representations.  
# 
# ## Step 7. Save the data
# 
# We prepared the the following variables for neural network:
# 
# 
# - *data_1*, *data_2*: padded numeric sequences for questions 1 and 2 in train sample 
# - *test_data_1*, *test_data_2*: padded numeric sequences for questions 1 and 2 in test sample
# - *nb_words*: length of dictionary *'word_index'* 
# - *embedding_matrix*: matrix for transformation in the embedding layer
# 
# Save these variables to *.pkl* files

# In[17]:


import pickle
with open('./savedData/data_1.pkl', 'wb') as f: pickle.dump(data_1, f, -1)
with open('./savedData/data_2.pkl', 'wb') as f: pickle.dump(data_2, f, -1)
with open('./savedData/nb_words.pkl', 'wb') as f: pickle.dump(nb_words, f, -1)
with open('./savedData/embedding_matrix.pkl', 'wb') as f: pickle.dump(embedding_matrix, f, -1)
with open('./savedData/test_data_1.pkl', 'wb') as f: pickle.dump(test_data_1, f, -1)
with open('./savedData/test_data_2.pkl', 'wb') as f: pickle.dump(test_data_2, f, -1)    


# The network will also use NLP features obtained using Spark in the first part of the project.
# 
# # Nework architecture
# 
# Quora released a [public dataset of duplicate questions](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) before the  competition, so, some interesting solutions had been already available before it started. 
# 
# Among them were approaches from:
# 
# - [Quora hackathon](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning), 
# - [Deep learning model](https://github.com/bradleypallen/keras-quora-question-pairs)  by Bradley Pallen et al.
# 
# Those approaches extensively use Recurrent Neural Networks usually with Long Short-Term Memory (LSTM) layers. 
# 
# The competition also showed the power of these methods. <br>
# 
# Participant [aphex34](https://www.kaggle.com/aphex34) was the only solo competitor among top 10 teams. He used NN techniques on all stages of reasearch including feature engeneering (see [7-th solution overview](https://www.kaggle.com/c/quora-question-pairs/discussion/34697#192676)). But his code has not been published.  
# 
# In order to solve the problem it is recommended to implement Ahmet Erdem's architecture which is relatively simple and also uses LSTM network.
# ![Ahmed LSTM](https://ilykei.com/api/fileProxy/documents%2FAdvanced%20Machine%20Learning%2FLecture%207%20AdvML%2FAhmedAlgo.png)
# 

# This network has 3 inputs: 
# 
# - input_1 and input_2 for questions converted to matrices (*data_1, data_2*) 
# - and input_3 for NLP features. 
# 
# Questions share the same embedding_1 and lstm_1 layers. 
# 
# Denote *y1* and *y2* outputs of layer *'lstm_1'* corresponding to the first and the second inputs, respectively.
# 
# Calculation inside red square is vector of squared differences of 2 outputs of layer *'lstm_1'*:
# 
# 1. Output *y1* is miltiplied by -1 in lambda_1 layer 
# 2. Then the result is added to *y2* in layer *'add_2'*. So, the output of layer *'add_2'* is difference between *y1* and *y2*. (Alternatively you can apply subtraction shown in Keras_basics.ipynb). 
# 3. The vector of differences is multiplied by itself element-wise in *'multiply_1'* layer. The result is vector of squared differences.
# 
# Then the vector of squared differences is concatenated in layer *'concatenate_1'* with sum of *y1* and *y2* obtained in layer *'add_1'*.  
# 
# The loss function to be minimized is *loss='binary_crossentropy'*.


# 
# 1. Prepare train and test data for network in local environment.
# 2. Implement the network above and tune it in local environment with part of the train data. <br>
#     Parameters to be tuned are: number of neurons in each layer, dropout rates (including recurrent_dropout of LSTM layer), standard deviation of *GaussianNoise* layer, *batch_size*. 
# 3. Run the model on the cluster with complete data and generate submission file as follows:
# 
# *submission = pd.DataFrame({"test_id": test_id, "is_duplicate": prediction_prob})*  
# *submission.to_csv("submission1.csv", index=False)*,
# 
# where *prediction_prob* is 1D array of prediction probabilities, *test_id* is index from *test_id* column of *test.csv* file.
# 
# Example of sbatch file to run task on GPU is given below (do not forget to remove end of line symbols < br > at the end of each line):

# In[20]:


# load pickle files
with open('savedData/data_1.pkl', 'rb') as f:
    data_1 = pickle.load(f)
with open('savedData/data_2.pkl', 'rb') as f:
    data_2 = pickle.load(f)
with open('savedData/nb_words.pkl', 'rb') as f: 
    nb_words = pickle.load(f)
with open('savedData/embedding_matrix.pkl', 'rb') as f: 
    embedding_matrix=pickle.load(f)
with open('savedData/test_data_1.pkl', 'rb') as f: 
    test_data_1=pickle.load(f)
with open('savedData/test_data_2.pkl', 'rb') as f: 
    test_data_2 = pickle.load(f) 


# In[23]:


# load features matrix from week 5
data_3 = np.loadtxt('train_features_1000.csv', delimiter=",", skiprows=1)
data_3.shape #1000x23

#data_1.shape # 1000x30
Ytrain=data_3[:,-1]


# remove last column of data_3
data_3=data_3[:,0:22]
data_3.shape


# In[ ]:


Ytrain[1:10,]


# In[21]:


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH=30
from keras.layers import Input, LSTM, BatchNormalization, GaussianNoise, concatenate, Lambda, add, Embedding, Dropout#,subtract
def getModel(lstmneurons=10, droprate=.1,  neurons1=10, neurons2=10, stdev=.1):
    # Three input layers
    input1 = Input(shape=(30,), name='input1')  # data_1: padded vector of 30 numbers
    input2 = Input(shape=(30,), name='input2') # data_2: padded vector of 30 numbers
    input3 = Input(shape=(22,), name='input3') # train_features_1000
    # embeddings layer
    embedding1 = Embedding(nb_words, EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(input1)
    embedding2 = Embedding(nb_words, EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(input2)
    # lstm layer
    lstm1 = LSTM(lstmneurons, activation='tanh', recurrent_activation='hard_sigmoid')(embedding1)
    # lstm layer
    lstm2 = LSTM(lstmneurons, activation='tanh', recurrent_activation='hard_sigmoid')(embedding2)
    # double the values
    add1 = add([lstm1, lstm2])
    # Create squared differences
    lambda1 = Lambda(lambda x: -1 * x)(lstm1)
    add2 = add([lstm2, lambda1])
    #add2=subtract([lstm1,lambda1])
    multiply1 = Lambda(lambda x: x**2)(add2)
    # combining squared differences and added values
    concatenate1 = concatenate([multiply1,add1], name='concatenate1') 
    # normalize the activations
    batch_normalization1 = BatchNormalization()(input3)
    # dense layer
    dense1 = Dense(neurons1, activation='relu', name='dense1')(batch_normalization1)
    # dropout - ignore portion of neuorns (regularization technique)
    dropout1 = Dropout(droprate, name='dropout1')(dense1)
    dropout2 = Dropout(droprate, name='dropout2')(concatenate1)
    # combine the two dropout layers
    concatenate2 = concatenate([dropout1,dropout2], name='concatenate2')
    # Normalize the activations
    batch_normalization2 = BatchNormalization()(concatenate2)
    # adding noise, zero centered
    gaussian_noise1 = GaussianNoise(stdev)(batch_normalization2)
    # dense layer
    dense2 = Dense(neurons2, activation='relu', name='dense2')(gaussian_noise1)
    # 3rd dropout layer
    dropout3 = Dropout(droprate, name='dropout3')(dense2)
    # Use softmax for output layer
    output1 = Dense(1, activation='sigmoid', name='output1')(dropout3)
    # define inputs and outputs for model
    model = Model(inputs=[input1,input2,input3], outputs=output1)
    # configure the model for training
    model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model



# In[25]:


# Use the function to create the model
model1=getModel()
# Train the model for a fixed number of epochs
model1.fit([data_1, data_2, data_3], Ytrain, epochs=200, batch_size=512,verbose=2,validation_split=0.2)


# In[26]:


predictions=model1.predict([test_data_1,test_data_2],512)


# In[ ]:


# def pro of one func
prob_of_one_udf = func.udf(lambda v: float(v[1]), FloatType())


# In[ ]:


outdf = predictions.withColumn('predict', func.round(prob_of_one_udf('probability'),6)).select('id','predict')
outdf.cache()
outdf.show(6)


# In[ ]:


# write csv
outdf.orderBy('id').coalesce(1).write.csv(outPath,header=True,mode='overwrite',quote="")


# In[ ]:


# assign batch size as variable
# lengths of vectrs we want to have
# global variabeldimension max sequence length(30)
# embedding layre needs embedding dimension ( 300)
# bigger batch size, smoother gradient, accurate search, but uses memory

#submission 1

#is_dubplicate    test_id
# select cutoff words

