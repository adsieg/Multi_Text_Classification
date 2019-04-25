Multi-classes task classification and LDA-based topic Recommender System
========================================================================

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
-   At character level -&gt; FastText
-   Topic Model as features // LDA features

#### LDA

Visualization provides a global view of the topics (and how they differ
from each other), while at the same time allowing for a deep inspection
of the terms most highly associated with each individual topic. A novel
method for choosing which terms to present to a user to aid in the task
of topic interpretation, in which we define the relevance of a term to a
topic.

![](https://github.com/adsieg/Multi_Text_Classification/blob/master/pictures/generative_LDA.gif)

![](https://github.com/adsieg/Multi_Text_Classification/blob/master/pictures/pyldavis.png)

![](https://github.com/adsieg/Multi_Text_Classification/blob/master/pictures/tsne_lda.png)

### C - Poincaré Embedding \[Embeddings and Hyperbolic Geometry\]

The main innovation here is that these embeddings are learnt in
**hyperbolic space**, as opposed to the commonly used **Euclidean
space**. The reason behind this is that hyperbolic space is more
suitable for capturing any hierarchical information inherently present
in the graph. Embedding nodes into a Euclidean space while preserving
the distance between the nodes usually requires a very high number of
dimensions.

<a href="https://arxiv.org/pdf/1705.08039.pdf" class="uri">https://arxiv.org/pdf/1705.08039.pdf</a>
<a href="https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Poincare%20Tutorial.ipynb" class="uri">https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Poincare%20Tutorial.ipynb</a>

**Learning representations** of symbolic data such as text, graphs and
multi-relational data has become a central paradigm in machine learning
and artificial intelligence. For instance, word embeddings such as
**WORD2VEC**, **GLOVE** and **FASTTEXT** are widely used for tasks
ranging from machine translation to sentiment analysis.

Typically, the **objective of embedding methods** is to organize
symbolic objects (e.g., words, entities, concepts) in a way such that
**their similarity in the embedding space reflects their semantic or
functional similarity**. For this purpose, the similarity of objects is
usually measured either by their **distance** or by their **inner
product** in the embedding space. For instance, Mikolov embed words in
*R*<sup>*d*</sup> such that their **inner product** is maximized when
words co-occur within similar contexts in text corpora. This is
motivated by the **distributional hypothesis**, i.e., that the meaning
of words can be derived from the contexts in which they appear.

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

5 - MyApp of multi-classes text classification with Attention mechanism
=======================================================================

![](https://github.com/adsieg/Multi_Text_Classification/blob/master/pictures/characters_attention.gif)

6 - Ressources / Bibliography
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

-   **\[Bonus\] Sentiment Analysis in PySpark** :
    <a href="https://github.com/tthustla/setiment_analysis_pyspark/blob/master/Sentiment%20Analysis%20with%20PySpark.ipynb" class="uri">https://github.com/tthustla/setiment_analysis_pyspark/blob/master/Sentiment%20Analysis%20with%20PySpark.ipynb</a>

-   **RNN Text Generation** :
    <a href="https://github.com/priya-dwivedi/Deep-Learning/blob/master/RNN_text_generation/RNN_project.ipynb" class="uri">https://github.com/priya-dwivedi/Deep-Learning/blob/master/RNN_text_generation/RNN_project.ipynb</a>

-   **Finding similar documents with Word2Vec and Soft Cosine Measure**:
    <a href="https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb" class="uri">https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb</a>

-   **\[!! ESSENTIAL !!\] Text Classification with Hierarchical
    Attention Networks**:
    <a href="https://humboldt-wi.github.io/blog/research/information_systems_1819/group5_han/" class="uri">https://humboldt-wi.github.io/blog/research/information_systems_1819/group5_han/</a>

-   **\[ESSENTIAL for any NLP Project\]**:
    <a href="https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks" class="uri">https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks</a>

-   **Doc2Vec + Logistic Regression** :
    <a href="https://github.com/susanli2016/NLP-with-Python/blob/master/Doc2Vec%20Consumer%20Complaint_3.ipynb" class="uri">https://github.com/susanli2016/NLP-with-Python/blob/master/Doc2Vec%20Consumer%20Complaint_3.ipynb</a>

-   **Doc2Vec -&gt; just embedding**:
    <a href="https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-wikipedia.ipynb" class="uri">https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-wikipedia.ipynb</a>

-   **New way of embedding -&gt; Poincaré Embeddings**:
    <a href="https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Poincare%20Tutorial.ipynb" class="uri">https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Poincare%20Tutorial.ipynb</a>

-   **Doc2Vec + Text similarity**:
    <a href="https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb" class="uri">https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb</a>

-   **Graph Link predictions + Part-of-Speech tagging tutorial with the
    Keras**:
    <a href="https://github.com/Cdiscount/IT-Blog/tree/master/scripts/link-prediction" class="uri">https://github.com/Cdiscount/IT-Blog/tree/master/scripts/link-prediction</a>
    &
    <a href="https://techblog.cdiscount.com/link-prediction-in-large-scale-networks/" class="uri">https://techblog.cdiscount.com/link-prediction-in-large-scale-networks/</a>

7. Other Topics - Text Similarity \[Word Mover Distance\]
=========================================================

-   **Finding similar documents with Word2Vec and WMD** :
    <a href="https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html" class="uri">https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html</a>

-   **Introduction to Wasserstein metric (earth mover’s distance)**:
    <a href="https://yoo2080.wordpress.com/2015/04/09/introduction-to-wasserstein-metric-earth-movers-distance/" class="uri">https://yoo2080.wordpress.com/2015/04/09/introduction-to-wasserstein-metric-earth-movers-distance/</a>

-   **Earthmover Distance**:
    <a href="https://jeremykun.com/2018/03/05/earthmover-distance/" class="uri">https://jeremykun.com/2018/03/05/earthmover-distance/</a>
    Problem: Compute distance between points with uncertain locations
    (given by samples, or differing observations, or clusters). For
    example, if I have the following three “points” in the plane, as
    indicated by their colors, which is closer, blue to green, or blue
    to red?

-   **Word Mover’s distance calculation between word pairs of two
    documents**:
    <a href="https://stats.stackexchange.com/questions/303050/word-movers-distance-calculation-between-word-pairs-of-two-documents" class="uri">https://stats.stackexchange.com/questions/303050/word-movers-distance-calculation-between-word-pairs-of-two-documents</a>

-   **Word Mover’s Distance (WMD) for Python**:
    <a href="https://github.com/stephenhky/PyWMD/blob/master/WordMoverDistanceDemo.ipynb" class="uri">https://github.com/stephenhky/PyWMD/blob/master/WordMoverDistanceDemo.ipynb</a>

-   \[LECTURES\] : **Computational Optimal Transport** :
    <a href="https://optimaltransport.github.io/pdf/ComputationalOT.pdf" class="uri">https://optimaltransport.github.io/pdf/ComputationalOT.pdf</a>

-   **Computing the Earth Mover’s Distance under Transformations** :
    <a href="http://robotics.stanford.edu/~scohen/research/emdg/emdg.html" class="uri">http://robotics.stanford.edu/~scohen/research/emdg/emdg.html</a>

-   **\[LECTURES\] Slides WMD**:
    <a href="http://robotics.stanford.edu/~rubner/slides/sld014.htm" class="uri">http://robotics.stanford.edu/~rubner/slides/sld014.htm</a>

Others \[Quora Datset\] :
-------------------------

-   **BOW + Xgboost Model** + **Word level TF-IDF + XgBoost** + **N-gram
    Level TF-IDF + Xgboost** + **Character Level TF-IDF + XGboost**:
    <a href="https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Xgboost_bow_tfidf.ipynb" class="uri">https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Xgboost_bow_tfidf.ipynb</a>

8 - Other Topics - Topic Modeling [LDA](#lda)
=============================================

<a href="https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb" class="uri">https://github.com/FelixChop/MediumArticles/blob/master/LDA-BBC.ipynb</a>

<a href="https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb" class="uri">https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb</a>

-   **TF-IDF + K-means & Latent Dirichlet Allocation (with Bokeh)**:
    <a href="https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html" class="uri">https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html</a>

-   **\[!! ESSENTIAL !!\] Building a LDA-based Book Recommender
    System**:
    <a href="https://humboldt-wi.github.io/blog/research/information_systems_1819/is_lda_final/" class="uri">https://humboldt-wi.github.io/blog/research/information_systems_1819/is_lda_final/</a>
