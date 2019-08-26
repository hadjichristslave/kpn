from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sentiment_analyzer import *


class Processor:

    def __init__(self):
        self.model_statistics = []

    def train_and_save_best_model(self, processed_data, processed_data_labels, vc: TfidfVectorizer):
        X_train, X_val, y_train, y_val = train_test_split(processed_data, processed_data_labels, train_size=TRAIN_TEST_SPLIT)
        self.train_all_models(X_train, X_val, y_train, y_val)

        best_model = self.calculate_best_model()

        print('best model was {0} with an f score of {1}'.format(best_model[0], best_model[1]))

        if SAVE_RESULTS:
            Helpers.save_model(best_model[0][0], file_path=FILE_PATH, model_name=SENTIMENT_ANALYZER_NAME)

        Helpers.print_most_significant_words(best_model[0], vc)

    def calculate_best_model(self):
        models_sorted_on_f_score = sorted(self.model_statistics, key=lambda x: x[1], reverse=True)
        best_model = models_sorted_on_f_score
        return best_model

    def train_all_models(self, X_train, X_val, y_train, y_val):
        for c in C_RANGE:
            # Linear models -> Scale better to big data
            model = LogisticRegression(C=c)
            self.fit_model_and_persist_fscore(X_train, X_val, c, model, y_train, y_val)
        for c in C_RANGE:
            model = LinearSVC(C=c)
            self.fit_model_and_persist_fscore(X_train, X_val, c, model, y_train, y_val)
        for c in BAYES_RANGE:
            model = MultinomialNB(alpha=c, fit_prior=True, class_prior=None)
            self.fit_model_and_persist_fscore(X_train, X_val, c, model, y_train, y_val)
        for c in FOREST_RANGE:
            model = RandomForestClassifier(n_estimators=c, random_state=42)
            self.fit_model_and_persist_fscore(X_train, X_val, c, model, y_train, y_val)
        for c in FOREST_RANGE:
            model = AdaBoostClassifier(n_estimators=c, random_state=42)
            self.fit_model_and_persist_fscore(X_train, X_val, c, model, y_train, y_val)
        for c in NEIGHBOR_RANGE:
            model = KNeighborsClassifier(n_neighbors=c)
            self.fit_model_and_persist_fscore(X_train, X_val, c, model, y_train, y_val)

    def fit_model_and_persist_fscore(self, X_train, X_val, c, model, y_train, y_val):
        try:
            model.fit(X_train, list(y_train))
            metrics = precision_recall_fscore_support(y_val, model.predict(X_val), average='micro')
            self.model_statistics.append([model, metrics[2]])
            print("PRf for %s C=%s: %s" % (str(type(model)), c, metrics))
        except Exception as ex:
            print("Failed to fit model {0} with exception {1}, continuing as normal".format(str(type(model)),ex))
