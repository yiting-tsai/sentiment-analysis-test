## LASER sentence embedding 
# - goal : multi-class classification
# - labels : `positive`, `negative`, `neutral`
# - accelerator : GPU for sentence embedding and TPU v3-8 model training
# - training time : 4 minutes for creating sentence embedding on GPU with LASER and 6 minutes DNN training on TPU

#### LASER Language-Agnostic SEntence Representations encoder
# > [FacebookResearch LASER](https://github.com/facebookresearch/LASER)
# > LASER encoder allow us to abstract away the language of the document by vectorizing the sentence with multilingual sentence embeddings. 
# > It was trained on 93 languages in 23 different alphabets, from major languages to minor dialects.
# > LASER was trained on sentences of at least 4 words; shorter sentences or words will have degraded performance.


#### model building process :
# Most processes are modularized into [functions](#hf). 

# 0. [import libraries, config TPU and set constant variables](#step0)
# 1. [load datasets in dataframe](#step1) `load_data_into_dataframe(train_csv_location, test_csv_location)`
# 2. [preprocess text to remove unwanted tokens such as @username or #hashtag](#step2) `preprocess_text(df)`
# 3. [convert dependent variable (categorical) to one-hot-encoding](#step3) `encode_dependent_variable_in_OHE(train_df, label_name='sentiment')`
# 4. [split training data further into training and validation set](#step4) `split_train_validation(train_df, X_columns_name, label, validation_size=0.15, random_state=42)`
# 5. [let LASER make sentence encoding](#step5)  `laser_encode(text, lang='en', normalize=True)`
# 6. Restart notebook and activate TPU, run all cells except the previous two on letting LASER make sentence embeddings and saving them, and load sentence embedding saved previsouly in `npy` format
# 7. [create neural network model in Keras](#step7) `build_model(num_classes=3, activation='softmax')')`
# 8. [set `EarlyStopping` by monitoring validation loss to prevent overfitting and `LearningRateScheduler` to schedule adaptive learning rate to speed up training time and hopefully increase performance](#step8)
# 9. [train model](#step9)
# 10. [plot model performance after training](#step10) `plot_model_history(history, measures)`
# 11. save model weights locally for inference

# Please refer to another notebook `data_analysis.ipynb` for data analysis. 

####### -------------------------------------------------------------------------------------- #######

### Import Libraries ### 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tqdm.keras import TqdmCallback

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow_addons as tfa  
from laserembeddings import Laser


### seeds ###
from numpy.random import seed
seed(42)
tf.random.set_seed(42)

### Constant Variable ###
TRAIN_CSV_PATH = '../input/technical-test/train.csv'
TEST_CSV_PATH = '../input/technical-test/test.csv'


### Helper Functions  ### 
# step 1. load datasets in dataframe
def load_data_into_dataframe(train_csv_location, test_csv_location):
    """
    Load CSV datasets into Pandas Dataframe.
    
    Parameters:
    ------------
    train_location : str
        A string of path location of training dataset in csv format.
    test_location : str 
        A string of path location of test dataset in csv format.
        
    Returns:
    ------------
    train_df : Pandas Dataframe
        A Dataframe of training data.
    test_df : Pandas Dataframe
        A Dataframe of test data.
    """
    # Load csv in Pandas Dataframe
    train_df = pd.read_csv(train_csv_location)
    test_df = pd.read_csv(test_csv_location)
    
    return train_df, test_df

# step 2. preprocess text to get rid of unwanted tokens such as @username or #hashtag
def preprocess_text(df):
    """
    Preprocess texts in content column of training data to remove unwanted tokens.
    """
    temp_df = pd.DataFrame(df.content.values)
    texts = temp_df[temp_df.columns.values[0]]
    texts = texts.apply(lambda s: re.sub('@\w+', ' ', s))            # remove @usernames
    texts = texts.apply(lambda s: re.sub('#',    ' ', s))            # remove hashtag prefixes    
    texts = texts.apply(lambda s: re.sub('\n',   ' ', s))            # remove newlines
    texts = texts.apply(lambda s: re.sub('\w+://\S+',  '<URL>', s))  # remove urls    
    texts = texts.apply(lambda s: re.sub('\s+',  ' ', s))            # remove multiple spaces    
    return list(texts)

# step 3. convert dependent variable (categorical) to one-hot-encoding
def encode_dependent_variable_in_OHE(train_df, label_name='sentiment', num_classes=3):
    """
    Encode the sentiment labels in the training Dataframe into one hot encoded format, since in the top layer output Dense layer use 'categorical_crossentropy'.
    
    Parameters:
    ------------
    train_df : Pandas Dataframe
        A Dataframe of loaded training data.
    label_name : str
        A string of column name indicating the output variable in the train_df.
    num_classes : int
        An integer indicating how many categories there are.
    
    Returns:
    ------------
    dummy_y : Numpy Array
        A matrix of one-hot-encoded class labels.
    """    
    # get array of sentiment labels 
    Y = train_df[label_name].values
    
    # encode label values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables - one hot encoded
    dummy_y = to_categorical(encoded_Y)
    
    return dummy_y

# step 4. split training data further into training and validation set
def split_train_validation(train_df, X_columns_name, label, validation_size=0.15, random_state=42):
    """
    Split training data into further training set and validation set in a ratio of ( 1-validation_size : validation_size).
    In my case, I set 85% of the training data for training the model and 15% aside for validation.
    
    Parameters:
    ------------
    train_df : Pandas Dataframe
        A Dataframe of loaded training data.
    X_columns_name : str
        A string of column name indicating the comments in texts in the train_df which will be used as independent variable for training.
    label : Numpy Array or Tensorflow Tensor
        An array of labels or a matrix of one-hot-encoded labels.
    validation_size : float
        A float number between 0-1 indicating desired validation size
    random_state : int
        Set the seed for reproucibility in splitting dataset.
    
    Returns:
    ------------
    train_texts : Numpy Array
        An array of training set samples in textual format in the shape of (1-validation_size,).
    val_texts : Numpy Array  
        An array of validation set samples in textual format in the shape of (validation_size,).
    y_train : Numpy Array  
        An array of 
    y_valid : Numpy Array 
       
    """   
    train_texts, val_texts, y_train, y_valid = train_test_split(train_df[X_columns_name].values, label, 
                                                  random_state=random_state, 
                                                  test_size=validation_size, shuffle=True)
    
    print('Total number of examples: ', len(train_df))
    print('number of training set examples: ', len(train_texts))
    print('number of validation set examples: ', len(val_texts))
    
    return train_texts, val_texts, y_train, y_valid

# step 5. Build LASER sentence embedding model to encode.
# based on https://www.kaggle.com/jamesmcguigan/nlp-laser-embeddings-keras
# normalize or not ? https://stats.stackexchange.com/questions/177905/should-i-normalize-word2vecs-word-vectors-before-using-them
def laser_encode(text, lang='en', normalize=True):
    """
    Build LASER embedding model to get ready to encode
    
    Parameters:
    ------------
    text : str or list of str
        A string or a list of string to be encoded by LASER.
    lang : str
        A string to specify which language to set the tokenizer, that string shall be in ISO 639-1 form.
    normalize : bool
        Default to True. Classification cares about the direction of vectors 
        in order to solve relations between vecotrs, so set normalize=True is more logical.
    
    Returns:
    ------------
    embedding : Numpy Array
        A matrix of encoded sentences in shape of (len([text]), 1024)
    """
    laser = Laser()
    
    if isinstance(text, str):
        sentences = [ text ]
    else:
        sentences = list(text)

    embedding = laser.embed_sentences(sentences, lang=lang)
    
    if normalize:
        embedding = embedding / np.sqrt(np.sum(embedding**2, axis=1)).reshape(-1,1)
        
    return embedding    

# step 7. create neural network model
def build_model(num_classes=3, activation='softmax'):
    """
    Create top layer on top of HuggingFace Transformer model for down-stream task. cls_token
    In my case, a multi-class classification is the goal. Taking into account that there are 3 classes, 
    I use categorical accuracy, as well as weighted F1 score and Matthews correlation coefficient as metrics.
    
    Parameters:
    ------------
    num_classes : int
        A integer representing num
    activation : str
        A string indicating which actvation to be used in the output layer. 
    
    Returns:
    ------------
    model : 
        configed model ready to be train
    """
    model = Sequential([
        Input(shape=(1024,)),
        Dense(units=512, activation=LeakyReLU(alpha=0.1)),
        Dense(units=256, activation=LeakyReLU(alpha=0.1)),
        Dropout(0.1),
        Dense(units=64, activation=LeakyReLU(alpha=0.1)),
        Dropout(0.2),
        Dense(units=32, activation=LeakyReLU(alpha=0.1)),
        Dense(units=num_classes, activation=activation, name='softmax')])
    
    # add weighted F1 score and Matthews correlation coefficient as metrics
    f1 = tfa.metrics.F1Score(num_classes=num_classes, average='weighted')
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=num_classes)
    
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy', f1, mcc])
    
    return model

# step 10. plot model performance after training
def plot_model_history(history, measures):
    """
    Plot history for visualization of performance measures in matplotlib.
    
    Parameters:
    ------------
    history : Keras History object
        A History object outputted from model.fit
    measure : str list 
        A list of string of which performance measures to be visualized    
    """
    for measure in measures:
        plt.plot(history.history[measure])
        plt.plot(history.history['val_' + measure])
        plt.title('model performance : ' + measure.replace("_", " "))
        plt.ylabel(measure.replace("_", " "))
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()    

####### -------------------------------------------------------------------------------------- #######

### 1. Load datasets in dataframe  ### 
train_df, test_df = load_data_into_dataframe(TRAIN_CSV_PATH, TEST_CSV_PATH)
# print(train_df.shape[0]) #--> 25000
# print(test_df.shape[0]) #--> 2500

# train_df.sentiment.value_counts()
# train_df.sentiment.value_counts() / # train_df.sentiment.value_counts().sum()

# remove the one example with 'unassigned' label (data analysis is done beforehand, so I'll just proceed to delete this example here)
train_df = train_df.drop(train_df[train_df.sentiment == 'unassigned'].index)
# train_df.head(10)


### 2. Preprocess text to remove unwanted tokens such as @username or #hashtag ###
preprocessed_content = preprocess_text(train_df)
train_df['preprocessed_content'] = preprocessed_content


### 3. Convert dependent variable (categorical) to one-hot-encoding ###
ohe_y = encode_dependent_variable_in_OHE(train_df, label_name='sentiment')


### 4. Split training data further into training and validation set ###
train_texts, val_texts, y_train, y_val = split_train_validation(train_df, 
                                                                X_columns_name='preprocessed_content', 
                                                                label=ohe_y, 
                                                                validation_size=0.15, 
                                                                random_state=42)


### 5. Let LASER make sentence encoding ###
import torch
print(torch.cuda.is_available())

# use GPU for speed up LASER embedding
x_train = laser_encode(train_texts.tolist(), lang='en', normalize=True)
x_val = laser_encode(val_texts.tolist(), lang='en', normalize=True)

# save sentence embedding in `npy`` format locally
with open('train.npy', 'wb') as f1:
    np.save(f1, x_train)
f1.close()    
with open('test.npy', 'wb') as f2:
    np.save(f2, x_val)
f2.close()


### 6. if use TPUs : reestart notebook and activate TPU, run all cells except the previous two on letting LASER 
#      make sentence embeddings and saving them, and load sentence embedding saved previsouly in npy format ###
with open('../input/laser-embedding/train.npy', 'rb') as f1:
    x_train = np.load(f1)
f1.close()    
with open('../input/laser-embedding/test.npy', 'rb') as f2:    
    x_val = np.load(f2)
f1.close()       


### TPU Config ###
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


### 7. Create neural network model ###
# load model onto TPU
#%%time
with strategy.scope():
    model = build_model(num_classes=3, activation='softmax')
model.summary()


### 8. Set `EarlyStopping` by monitoring validation loss to prevent overfitting and `LearningRateScheduler`
#      to schedule adaptive learning rate to speed up training time and hopefully increase performance  ###
EPOCHS = 100
BATCH_SIZE = 32 * strategy.num_replicas_in_sync

ES_callback = EarlyStopping(monitor='val_loss', patience=3, mode='auto')

def scheduler(epoch, lr):
    if epoch < 60:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

LR_callback = LearningRateScheduler(scheduler) 


### 9. Train model ###
n_steps = train_texts.shape[0] // BATCH_SIZE

train_history = model.fit(
    x_train, y_train,
    steps_per_epoch=n_steps,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    callbacks=[TqdmCallback(verbose=2), ES_callback, LR_callback]
)

### 10. Plot model performance after training ###
plot_model_history(train_history, ['categorical_accuracy', 'loss', 'f1_score'])


### 11. Save model weights locally for inference ###
model.save('laser-DNN_model.h5')