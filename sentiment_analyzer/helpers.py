import pickle
import re

from nltk.stem.api import StemmerI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from main import preprocessor, processor

characters_to_disregard = re.compile("[.;:!\'?,\"()\[\]]")


class Helpers:

    @staticmethod
    def extract_rating(x) -> int:
        return int(x.split()[0])

    @staticmethod
    def normalize_text(raw_text: str, stemmer: StemmerI) -> str:
        if raw_text == '':
            return ''
        character_stripped_text = ''.join([characters_to_disregard.sub("", line.lower()) for line in raw_text])
        normalized_text = ' '.join([stemmer.stem(word) for word in character_stripped_text.split()])
        return normalized_text

    @staticmethod
    def save_model(model, file_path, model_name):
        pickle.dump(model, open('{0}{1}'.format(file_path, model_name), 'wb'))

    @staticmethod
    def load_model(file_path, model_name):
        loaded_model = pickle.load(open('{0}{1}'.format(file_path, model_name), 'rb'))
        return loaded_model

    @staticmethod
    def print_most_significant_words(final_tfidf: LinearSVC, vc: TfidfVectorizer):
        feature_to_coef = {word: coef for word, coef in zip(vc.get_feature_names(), final_tfidf[0].coef_[0])}
        Helpers.generate_plot_word_cloud_for_weighted_terms(terms=sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:60])
        Helpers.generate_plot_word_cloud_for_weighted_terms(terms=sorted(feature_to_coef.items(), key=lambda x: x[1])[:60])

    @staticmethod
    def generate_plot_word_cloud_for_weighted_terms(terms):
        weighted_text = ' '.join([(term[0].replace(' ', '_') + ' ') * abs(int(term[1] * 10)) for term in terms])
        wordcloud = WordCloud(max_font_size=20, prefer_horizontal=.4, width=320, height=240,
                              max_words=50,  background_color="white",
                              collocations=False).generate(weighted_text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


def train_on_data(data_to_train):
    x_aug, y_aug = preprocessor.over_sample_data(data_to_train)
    processor.train_and_save_best_model(processed_data=x_aug, processed_data_labels=y_aug, vc=preprocessor.tf_idf_vectorizer)