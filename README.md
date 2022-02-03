- Synthesio technical test
  - [✋ How to get inference ?](#user-content--how-to-get-inference)
  - [📚 Methodology](#user-content--methodology)
  - [👓 Addition reading](#user-content--additional-reading)
  - [🌳 Repo tree](#user-content--repo-tree)



## ✋ How to get inference ?
1. clone this repo `git clone https://github.com/yiting-tsai/sentiment-analysis-test.git`
2. get model weights from [google drive download link](https://drive.google.com/file/d/1zlsLILAYa_nekjEQ0VqstZ_2nX2iOqTI/view?usp=sharing) (logged in google account required)
    * because the model weight file exceeds 1GB and Git LFS doesn't support pushing LFS objects to public forks ([issue src link](https://github.com/git-lfs/git-lfs/issues/1906#issuecomment-276602035))
3. go to this repo and place the downloaded `tf_model.h5` to the folder of this repo `./model_weights_config/xlmr-model/`
4. create a virtual environment `virtualenv myEnv` and active `source myEnv/bin/activate`
5. install necessary python packages `pip install -r requirements`
6. run on terminal actual inference script `python xlmr-inference.py` (apprx. 10 mins to finish if on CPU)
7. inference results are saved in `predictions.csv` in the repo


## 📚 Methodology
1. Understand the goal - multi-class classification on user-generated short comments in multiple languages
2. Understand the data - user-generated short comments in various lengths and in 17+ langugages
    - 👐 *please refer to data analytic notebook in `notebook/analytics/data_analytics.ipynb` for more details*
3. Understand the literature - multilingual embeddings (mBERT, XLM, XLM-RoBERTa, LASER, etc) via reading survey papers (ie [survey paper](https://arxiv.org/abs/2107.00676))
4. Modeling - two models : **XLM-R** (transfer learning) & **LASER + neural networks** (sentence embedding + classification model)
    - build the model 
    - choose `softmax` as activation on final layer for multi-class classification, 
    - choose ideal optimizer Adam, computationally efficient and handling well sparse grad on noisy problems
    - choose metrics accuracy, weighted f1 and Matthew correlation coefficients
    - set callbacks like early stopping to stop the model training if validation loss increases after 3 epochs in order to prevent overfitting on training data
    - efficient training on accelerators GPU and TPU on kaggle notebook
    - 👐 *please refer to individual train notebook/script of two models for more details*
5. Model analysis - black-box analysis on the performance resutls of the model
  - additional baseline model for comparison: zero-shot classification of XLM-R
    1. acc 0.52, F1 0.456, MCC 0.339
    2. both XLM-R and LASER+NN perform better than baseline model
  - **XLM-R**
    1. accurarcy reachs more than 80% within only 10 epochs
    2. posssible improvement:
      - add more layers to the down-stream task (I only use one layer)
      - increase data size
      - add text preprocessing to remove stopwords, punctuations or unnecessary tokens like `@username` or `#hashtag` (also removing `@username` for anonymization / privacy concern)
    3. XLM-R tokenizes with SentencePiece (language independent and data driven), which could be one of driven factors to its performance 
  - **LASER + NN**
    1. accurarcy reachs 80% after 50 epochs
    2. LASER performance degrades for shorter sentence because it was trained on sentences of at least 4 words [ref](https://github.com/facebookresearch/LASER/issues/44), and 15% of texts in training data is less than 25 characters, rather short sentence
    3. possible improvement:
      - specify on language when tokenizing for LASER with detected language info
      - apply batch normalization between layers in the NN
      - train more epochs
      - increase data size


### 👓 Additional reading
1. XLM-R [paper](https://arxiv.org/abs/1911.02116)  /  [FacebookReseach blog](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/)
2. LASER [repo](https://github.com/facebookresearch/LASER)
3. Multilingual word embeddings 
  - [HuggingFace](https://huggingface.co/docs/transformers/multilingual)  
  - [A Primer on Pretrained Multilingual Language Models](https://arxiv.org/abs/2107.00676)  
  - [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)


## 🌳 Repo tree
```python
├── data
│   ├── test.csv
│   └── train.csv
├── LICENSE
├── model_weights_config
│   ├── laser-dnn-model
│   │   ├── laser_model.h5
│   │   ├── test_embedding.npy
│   │   └── train_embedding.npy
│   └── xlmr-model
│       ├── config.json
│       └── softmax.pickle
├── notebook
│   ├── analytics
│   │   ├── data
│   │   │   ├── analyzed_train.csv
│   │   │   └── lang_detect.csv
│   │   ├── data_analytics.ipynb
│   │   ├── model
│   │   │   ├── laser_dnn_history.json
│   │   │   └── xlmr_history.json
│   │   └── model_analytics.ipynb
│   ├── inference
│   │   └── xlmr-inferenc_.ipynb
│   └── train
│       ├── laser-DNN_train_kaggle.ipynb
│       └── xlmr_train_kaggle.ipynb
├── predictions.csv
├── README.md
├── requirements.txt
├── script
│   ├── train_laser-DNN_kaggle.py
│   └── train_xlmr_kaggle.py
├── test-instructions.md
└── xlmr-inference.py
```
