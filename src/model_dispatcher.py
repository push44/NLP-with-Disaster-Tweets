from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer

models = {
    "logistic_reg": LogisticRegression(max_iter=10000, C=0.6),
    "naive_bayes": naive_bayes.MultinomialNB(),
    "tfidf_vec": TfidfVectorizer(ngram_range=(1,3),max_features=1000),
}