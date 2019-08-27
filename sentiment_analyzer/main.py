from sentiment_analyzer.preprocessor import Preprocessor
from sentiment_analyzer.processor import Processor
from sentiment_analyzer import *
import warnings
warnings.filterwarnings("ignore")


def train_and_save_new_model(data_to_train):
    x_aug, y_aug = preprocessor.over_sample_data(data_to_train)
    processor.train_and_save_best_model(processed_data=x_aug, processed_data_labels=y_aug, vc=preprocessor.tf_idf_vectorizer)

if __name__ == '__main__':
    preprocessor = Preprocessor()
    processor = Processor()
    preprocessor.load_and_preprocess_data(path='data/training_data/raw_dataset.csv')


    if USE_SVD:
        train_and_save_new_model(preprocessor.clean_reduced_train_data)
    else:
        train_and_save_new_model(preprocessor.clean_train_data)

