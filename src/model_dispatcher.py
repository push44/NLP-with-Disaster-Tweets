from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes

models = {
    "logistic_reg": LogisticRegression(max_iter=10000, C=0.6),
    "naive_bayes": naive_bayes.MultinomialNB()
}