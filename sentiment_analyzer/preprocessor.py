from sentiment_analyzer import *
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt


class Preprocessor:
    def __init__(self):
        self.path = ''
        self.data: pd.DataFrame = pd.DataFrame
        self.stemmer = PorterStemmer()
        self.tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords.words('english'))
        self.svd = TruncatedSVD(algorithm='randomized', n_components=2000, n_iter=7, random_state=42, tol=0.0)
        self.clean_train_data = []
        self.clean_reduced_train_data = []

    def load_and_preprocess_data(self, path):
        self.data = pd.DataFrame().from_csv(path)
        self.data.set_index('id', inplace=True)
        self.data['clean_rating'] = self.data['score'].apply(Helpers.extract_rating)
        self.data['title'] = self.data['title'].apply(Helpers.normalize_text, stemmer=self.stemmer)
        self.data['text'] = self.data['text'].apply(Helpers.normalize_text, stemmer=self.stemmer)
        self.tf_idf_vectorizer.fit(self.data['text'])
        train_ready_data = list(self.data['text'] + ' ' + self.data['title'])
        self.clean_train_data = self.tf_idf_vectorizer.transform(train_ready_data)
        self.clean_reduced_train_data = self.svd.fit_transform(self.clean_train_data)

        if SAVE_RESULTS:
            Helpers.save_model(self.tf_idf_vectorizer, file_path=FILE_PATH, model_name=TF_IDF_VECTORIZER_NAME)


    def over_sample_data(self, data_to_oversample):
        self.plot_initial_data_distribution()
        sm = SMOTE(random_state=42, sampling_strategy='auto')
        X_res, y_res = sm.fit_resample(data_to_oversample, list(self.data['clean_rating']))
        return X_res, y_res

    def plot_initial_data_distribution(self):
        percents = self.data.groupby('clean_rating').count() / self.data.shape[0]
        labels = ['Bad', 'Poor', 'Average', 'Good', 'Excelent']
        sizes = list(percents['title'].apply(lambda x: round(x * 100, 2)))
        colors = ['red', 'orange', 'yellow', 'yellowgreen', 'green']
        explode = [0.1] * 5
        patches = plt.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode, labels=labels)
        plt.legend(patches[0], labels, loc="best")
        plt.title("Distribution of Rebel's reviews ratings on a total of {0} reports".format(self.data.shape[0]))
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.tight_layout()
        plt.show()




