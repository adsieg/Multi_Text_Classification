Here is **my winning strategy** to carry multi-text classification task
out.

**Data Source** :
<a href="https://catalog.data.gov/dataset/consumer-complaint-database" class="uri">https://catalog.data.gov/dataset/consumer-complaint-database</a>

1 - Text Mining
===============

-   **Word Frequency Plot**: Compare frequencies across different texts
    and quantify how similar and different these sets of word
    frequencies are using a correlation test. How correlated are the
    word frequencies between text1 and text2, and between text1 and
    text3?

![](https://github.com/adsieg/Multi_Text_Classification/blob/master/pictures/word_frequency.png)

-   **Most discriminant and important word per categories**

-   **Relationships between words & Pairwise correlations**: examining
    which words tend to follow others immediately, or that tend to
    co-occur within the same documents.

Which word is associated with another word? Note that this is a
visualization of a Markov chain, a common model in text processing. In a
Markov chain, each choice of word depends only on the previous word. In
this case, a random generator following this model might spit out
“collect”, then “agency”, then “report/credit/score”, by following each
word to the most common words that follow it. To make the visualization
interpretable, we chose to show only the most common word to word
connections, but one could imagine an enormous graph representing all
connections that occur in the text.

-   **Distribution of words**: Want to show that there are similar
    distributions for all texts, with many words that occur rarely and
    fewer words that occur frequently. Here is the goal of Zip Law
    (extended with Harmonic mean) - Zipf’s Law is a statistical
    distribution in certain data sets, such as words in a linguistic
    corpus, in which the frequencies of certain words are inversely
    proportional to their ranks.

![](https://github.com/adsieg/Multi_Text_Classification/blob/master/pictures/word_correlations.png)

-   **How to spell variants of a given word**

-   **Chi-Square to see which words are associated to each category**:
    find the terms that are the most correlated with each of the
    categories

-   **Part of Speech Tags** and **Frequency distribution of POST**: Noun
    Count, Verb Count, Adjective Count, Adverb Count and Pronoun Count

-   **Metrics of words**: *Word Count of the documents* – ie. total
    number of words in the documents, *Character Count of the documents*
    – total number of characters in the documents, *Average Word Density
    of the documents* – average length of the words used in the
    documents, *Puncutation Count in the Complete Essay* – total number
    of punctuation marks in the documents, *Upper Case Count in the
    Complete Essay* – total number of upper count words in the
    documents, *Title Word Count in the Complete Essay* – total number
    of proper case (title) words in the documents

2 - Word Embedding
==================

### A - Frequency Based Embedding

-   Count Vector
-   TF IDF
-   Co-Occurrence Matrix with a fixed context window (SVD)
-   TF-ICF
-   Function Aware Components

### B - Prediction Based Embedding

-   CBOW (word2vec)
-   Skip-Grams (word2vec)
-   Glove
-   Topic Model as features // LDA features

3 - Algorithms
==============

### A - Traditional Methods

-   CountVectorizer + Logistic
-   CountVectorizer + NB
-   CountVectorizer + LightGBM
-   HasingTF + IDF + Logistic Regression
-   TFIDF + NB
-   TFIDF + LightGBM
-   TF-IDF + SVM
-   Hashing Vectorizer + Logistic
-   Hashing Vectorizer + NB
-   Hashing Vectorizer + LightGBM
-   Bagging / Boosting
-   Word2Vec + Logistic
-   Word2Vec + LightGNM
-   Word2Vec + XGBoost
-   LSA + SVM

### B - Deep Learning Methods

-   GRU + Attention Mechanism
-   CNN + RNN + Attention Mechanism
-   CNN + LSTM/GRU + Attention Mechanism

4 - Explainability
==================

**Goal**: explain predictions of arbitrary classifiers, including text
classifiers (when it is hard to get exact mapping between model
coefficients and text features, e.g. if there is dimension reduction
involved)

-   Lime
-   Skate
-   Shap

![](https://github.com/adsieg/Multi_Text_Classification/blob/master/pictures/explainability.gif)

5 - Ressources / Bibliography
=============================

-   **All models** :
    <a href="https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/" class="uri">https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/</a>

-   **CNN Text Classification**:
    <a href="https://github.com/cmasch/cnn-text-classification/blob/master/Evaluation.ipynb" class="uri">https://github.com/cmasch/cnn-text-classification/blob/master/Evaluation.ipynb</a>

-   **CNN Multichannel Text Classification + Hierarchical attention +
    …**:
    <a href="https://github.com/gaurav104/TextClassification/blob/master/CNN%20Multichannel%20Text%20Classification.ipynb" class="uri">https://github.com/gaurav104/TextClassification/blob/master/CNN%20Multichannel%20Text%20Classification.ipynb</a>

-   **Notes for Deep Learning**
    <a href="https://arxiv.org/pdf/1808.09772.pdf" class="uri">https://arxiv.org/pdf/1808.09772.pdf</a>

-   **Doc classification with NLP**
    <a href="https://github.com/mdh266/DocumentClassificationNLP/blob/master/NLP.ipynb" class="uri">https://github.com/mdh266/DocumentClassificationNLP/blob/master/NLP.ipynb</a>

-   **Paragraph Topic Classification**
    <a href="http://cs229.stanford.edu/proj2016/report/NhoNg-ParagraphTopicClassification-report.pdf" class="uri">http://cs229.stanford.edu/proj2016/report/NhoNg-ParagraphTopicClassification-report.pdf</a>

-   **1D convolutional neural networks for NLP**
    <a href="https://github.com/Tixierae/deep_learning_NLP/blob/master/cnn_imdb.ipynb" class="uri">https://github.com/Tixierae/deep_learning_NLP/blob/master/cnn_imdb.ipynb</a>

-   **Hierarchical Attention for text classification**
    <a href="https://github.com/Tixierae/deep_learning_NLP/blob/master/HAN/HAN_final.ipynb" class="uri">https://github.com/Tixierae/deep_learning_NLP/blob/master/HAN/HAN_final.ipynb</a>

-   **Multi-class classification scikit learn** (Random forest, SVM,
    logistic regression)
    <a href="https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f" class="uri">https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f</a>
    <a href="https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb" class="uri">https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb</a>

-   **Text feature extraction TFIDF mathematics**
    <a href="https://dzone.com/articles/machine-learning-text-feature-0" class="uri">https://dzone.com/articles/machine-learning-text-feature-0</a>

-   **Classification Yelp Reviews (AWS)**
    <a href="http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/" class="uri">http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/</a>

-   **Convolutional Neural Networks for Text Classification (waouuuuu)**
    <a href="http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/" class="uri">http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/</a>
    <a href="https://github.com/davidsbatista/ConvNets-for-sentence-classification" class="uri">https://github.com/davidsbatista/ConvNets-for-sentence-classification</a>

-   **3 ways to interpretate your NLP model \[Lime, ELI5, Skater\]**
    <a href="https://github.com/makcedward/nlp/blob/master/sample/nlp-model_interpretation.ipynb" class="uri">https://github.com/makcedward/nlp/blob/master/sample/nlp-model_interpretation.ipynb</a>
    <a href="https://towardsdatascience.com/3-ways-to-interpretate-your-nlp-model-to-management-and-customer-5428bc07ce15" class="uri">https://towardsdatascience.com/3-ways-to-interpretate-your-nlp-model-to-management-and-customer-5428bc07ce15</a>
    <a href="https://medium.freecodecamp.org/how-to-improve-your-machine-learning-models-by-explaining-predictions-with-lime-7493e1d78375" class="uri">https://medium.freecodecamp.org/how-to-improve-your-machine-learning-models-by-explaining-predictions-with-lime-7493e1d78375</a>

-   **Deep Learning for text made easy with AllenNLP**
    <a href="https://medium.com/swlh/deep-learning-for-text-made-easy-with-allennlp-62bc79d41f31" class="uri">https://medium.com/swlh/deep-learning-for-text-made-easy-with-allennlp-62bc79d41f31</a>

-   **Ensemble Classifiers**
    <a href="https://www.learndatasci.com/tutorials/predicting-reddit-news-sentiment-naive-bayes-text-classifiers/" class="uri">https://www.learndatasci.com/tutorials/predicting-reddit-news-sentiment-naive-bayes-text-classifiers/</a>

-   **Classification Algorithms ** \[tfidf, count features, logistic
    regression, naive bayes, svm, xgboost, grid search, word vectors,
    LSTM, GRU, Ensembling\] :
    <a href="https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle/notebook" class="uri">https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle/notebook</a>

-   **Deep learning architecture** \[TextCNN, BiDirectional
    RNN(LSTM/GRU), Attention Models\] :
    <a href="https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/" class="uri">https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/</a>
    and
    <a href="https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras" class="uri">https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras</a>

-   **CNN + Word2vec and LSTM + Word2Vec** :
    <a href="https://www.kaggle.com/kakiac/deep-learning-4-text-classification-cnn-bi-lstm" class="uri">https://www.kaggle.com/kakiac/deep-learning-4-text-classification-cnn-bi-lstm</a>

-   **Comparison of models** \[Bag of Words - Countvectorizer Features,
    TFIDF Features, Hashing Features, Word2vec Features\] :
    <a href="https://mlwhiz.com/blog/2019/02/08/deeplearning_nlp_conventional_methods/" class="uri">https://mlwhiz.com/blog/2019/02/08/deeplearning_nlp_conventional_methods/</a>

-   **Embed, encode, attend, predict** :
    <a href="https://explosion.ai/blog/deep-learning-formula-nlp" class="uri">https://explosion.ai/blog/deep-learning-formula-nlp</a>

-   Visualisation sympa pour comprendre CNN :
    <a href="http://www.thushv.com/natural_language_processing/make-cnns-for-nlp-great-again-classifying-sentences-with-cnns-in-tensorflow/" class="uri">http://www.thushv.com/natural_language_processing/make-cnns-for-nlp-great-again-classifying-sentences-with-cnns-in-tensorflow/</a>

-   **Yelp comments classification \[ LSTM, LSTM + CNN\]** :
    <a href="https://github.com/msahamed/yelp_comments_classification_nlp/blob/master/word_embeddings.ipynb" class="uri">https://github.com/msahamed/yelp_comments_classification_nlp/blob/master/word_embeddings.ipynb</a>

-   **RNN text classification** :
    <a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/" class="uri">https://karpathy.github.io/2015/05/21/rnn-effectiveness/</a>

-   **CNN for Sentence Classification** & **DCNN for Modelling
    Sentences** & **VDNN for Text Classification** & **Multi Channel
    Variable size CNN** & **Multi Group Norm Constraint CNN** & **RACNN
    Neural Networks for Text Classification**:
    <a href="https://bicepjai.github.io/machine-learning/2017/11/10/text-class-part1.html" class="uri">https://bicepjai.github.io/machine-learning/2017/11/10/text-class-part1.html</a>

-   **Transformers** :
    <a href="https://towardsdatascience.com/transformers-141e32e69591" class="uri">https://towardsdatascience.com/transformers-141e32e69591</a>

-   **Seq2Seq** :
    <a href="https://guillaumegenthial.github.io" class="uri">https://guillaumegenthial.github.io</a>
    /sequence-to-sequence.html

-   **The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer
    Learning)** :
    <a href="https://jalammar.github.io/" class="uri">https://jalammar.github.io/</a>

-   **LSTM & GRU explanation** :
    <a href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21" class="uri">https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21</a>

-   **Text classification using attention mechanism in Keras** :
    <a href="http://androidkt.com/text-classification-using-attention-mechanism-in-keras/" class="uri">http://androidkt.com/text-classification-using-attention-mechanism-in-keras/</a>

-   **Bernoulli Naive Bayes & Multinomial Naive Bayes & Random Forests &
    Linear SVM & SVM with non-linear kernel**
    <a href="https://github.com/irfanelahi-ds/document-classification-python/blob/master/document_classification_python_sklearn_nltk.ipynb" class="uri">https://github.com/irfanelahi-ds/document-classification-python/blob/master/document_classification_python_sklearn_nltk.ipynb</a>
    and
    <a href="https://richliao.github.io/" class="uri">https://richliao.github.io/</a>

-   **DL text classification** :
    <a href="https://gitlab.com/the_insighters/data-university/nuggets/document-classification-with-deep-learning" class="uri">https://gitlab.com/the_insighters/data-university/nuggets/document-classification-with-deep-learning</a>

-   **1-D Convolutions over text** :
    <a href="http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/" class="uri">http://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/</a>
    and
    <a href="https://github.com/davidsbatista/ConvNets-for-sentence-classification/blob/master/Convolutional-Neural-Networks-for-Sentence-Classification.ipynb" class="uri">https://github.com/davidsbatista/ConvNets-for-sentence-classification/blob/master/Convolutional-Neural-Networks-for-Sentence-Classification.ipynb</a>

-   \[Bonus\] Sentiment Analysis in PySpark :
    <a href="https://github.com/tthustla/setiment_analysis_pyspark/blob/master/Sentiment%20Analysis%20with%20PySpark.ipynb" class="uri">https://github.com/tthustla/setiment_analysis_pyspark/blob/master/Sentiment%20Analysis%20with%20PySpark.ipynb</a>
