### Import Libraries ###
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow_addons as tfa 
import transformers
from transformers import TFAutoModel, AutoTokenizer


### Check GPU ###
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


### Config / Set Constant Variable ###
TEST_CSV_PATH = './data/test.csv'

BATCH_SIZE = 16
MAX_LEN = 256
MODEL_NAME = 'jplu/tf-xlm-roberta-base'
MODEL_WEIGHT_CONFIG_DIR ='./model_weights_config/xlmr-model/'

os.environ["TOKENIZERS_PARALLELISM"] = 'false'


### Set Tokenizer ###
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


### Helper Functions ###
# load test set into Pandas Dataframe
def load_test_data_into_dataframe(test_csv_location):
    """
    Load CSV datasets into Pandas Dataframe.
    
    Parameters:
    ------------
    test_location : str 
        A string of path location of test dataset in csv format.
        
    Returns:
    ------------
    test_df : Pandas Dataframe
        A Dataframe of test data.
    """
    # Load csv in Pandas Dataframe
    test_df = pd.read_csv(test_csv_location)
    
    return test_df

# Transformer Tokenizer tokenize and encode the textual data into embedding
def tokenizer_encode(texts, tokenizer, maxlen=512):
    """
    Let Transformers Tokenizer API prepare the input data and encode, precisely tokenizing 
    
    Parameters:
    ------------
    texts : list of str
        A list of string to be encoded by the tokenizer.
    tokenizer : Transformers AutoTokenizer
        A Tensorflow AutoTokenizer object loaded in order to encode the text data.
    max_len : int
        An integer representing the maximun length of each sample, also as the shape of outputs from 'frozen' body of transformer model.
        
    Returns:
    ------------
    model : Numpy Array
        An array of tokenizer-encoded vector from the texts.
    """
    encoding = tokenizer.batch_encode_plus(
        texts,
        truncation=True,
        return_attention_mask=False, 
        return_token_type_ids=False,
        padding='max_length',
        max_length=maxlen
    )
    
    encoding_array = np.array(encoding['input_ids'])
    
    return encoding_array

# load test set into Tensorflow Dataset API
def load_test_into_tf_Dataset(tokenizer, test_df, batch_size=BATCH_SIZE):
    """
    Load splitted test dataset into Tensorflow Dataset API for a more efficient input pipeline, especially for parallelism.
    
    Parameters:
    ------------
    tokenizer : Transformers AutoTokenizer
        A Tensorflow AutoTokenizer object loaded in order to encode the text data.
    test_df : Pandas Dataframe
        A Dataframe of loaded test data.
    batch_size : int
        An integer indicating the size of the batch. Here uses 16*num_of_TPU_core (=128) by default.

    
    Returns 
    ------------
    test_dataset : tf.data.Dataset
        A Tensorflow Dataset API object of test set as an input pipeline for model inference.
    """
    ## Tokenize the textual format data by calling tokenizer_encode()
    x_test = tokenizer_encode(texts=test_df.content.values.tolist(), tokenizer=tokenizer, maxlen=MAX_LEN)
    
    ## Build Tensorflow Dataset objects
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCH_SIZE)
    )
    
    return test_dataset 

# build output layer on top of the transformer model
def build_model(transformer, num_classes=3, activation='softmax', max_len=512):
    """
    Create top layer on top of HuggingFace Transformer model for down-stream task. cls_token
    In my case, a multi-class classification is the goal. Taking into account that there are 3 classes, 
    I use categorical accuracy, as well as weighted F1 score and Matthews correlation coefficient as metrics.
    
    Parameters:
    ------------
    transformer : Transformers TFAutoModel
        A string of path location of training dataset in csv format.
    num_classes : int
        A integer representing num
    activation : str
        A string indicating which actvation to be used in the output layer. 
    max_len : int
        An integer representing the maximun length of each sample, also as the shape of outputs from 'frozen' body of transformer model.
        
    Returns:
    ------------
    model : 
        configed model ready to be used
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(units=num_classes, activation=activation, name=activation)(cls_token) # set units=3 because we have three classes
    
    # add weighted F1 score and Matthews correlation coefficient as metrics
    f1 = tfa.metrics.F1Score(num_classes=num_classes, average='weighted')
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=num_classes)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy', f1, mcc])
    
    return model

# load model weights from fine-tuned model weights saved in directory
def load_model(model_dir=MODEL_WEIGHT_CONFIG_DIR, max_len=MAX_LEN):
    """
    Function to load a keras model that uses a transformer layer
    
    Parameters :
    ------------
    model_dir : str
        A string indicating where model's weight and config file are.
    max_len : int
        An integer representing the maximun length of each sample, to be passed to build_model() function.

    Returns:
    ------------
    model : 
        configed model with weights loaded from fine-tuned model.
    """
    transformer = TFAutoModel.from_pretrained(model_dir)
    model = build_model(transformer, max_len=max_len)
    softmax = pickle.load(open(model_dir+'softmax.pickle', 'rb'))
    model.get_layer('softmax').set_weights(softmax)

    return model

# function to be used in later coonverting inference label from int back to string
def label_int_2_str(x): 
    """
    Convert encoded int labels back to string sentiment labels.
    """
    if x == 0:
        return 'negative'
    elif x == 1:
        return 'neutral'
    elif x == 2:
        return 'positive'


### Load Data ###
# load test set into Pandas Dataframe and Tensorflow Dataset API
test_df = load_test_data_into_dataframe(TEST_CSV_PATH)
test_dataset = load_test_into_tf_Dataset(tokenizer=tokenizer, test_df=test_df, batch_size=BATCH_SIZE)


### Load Model & Inference ###
if len(tf.config.list_physical_devices('GPU')) > 0:
    # place proces on GPU
    print('Using GPU for inference.\n')
    with tf.device('/GPU:0'):
        # load model weights from fine-tuned model weights saved in /xlmr-model/ directory
        model = load_model(model_dir=MODEL_WEIGHT_CONFIG_DIR, max_len=MAX_LEN)
        # inference
        prediction_array = model.predict(test_dataset, verbose=1)
elif len(tf.config.list_physical_devices('GPU')) == 0:
    # place process on CPU
    print('Using CPU for inference.\n')
    with tf.device('/CPU:0'):
        # load model weights from fine-tuned model weights saved in /xlmr-model/ directory
        model = load_model(model_dir=MODEL_WEIGHT_CONFIG_DIR, max_len=MAX_LEN)
        # inference
        prediction_array = model.predict(test_dataset, verbose=1)


# convert probability for each class to int label
test_df['inference_int'] = np.argmax(prediction_array, axis=-1)
# convert int label to string label : 0=negative, 1=neutral, 2=positive
test_df['inference_sentiment'] = test_df['inference_int'].apply(label_int_2_str)
# drop column of int label
test_df = test_df.drop('inference_int', axis=1)
# save inference results to `predictions.csv`
test_df.to_csv('predictions.csv')