from helpers import train_on_data
from sentiment_analyzer.preprocessor import Preprocessor
from sentiment_analyzer.processor import Processor

if __name__ == '__main__':
    preprocessor = Preprocessor()
    processor = Processor()
    preprocessor.load_and_preprocess_data(path='data/training_data/raw_dataset.csv')

    # train_on_data(preprocessor.clean_train_data)
    train_on_data(preprocessor.clean_reduced_train_data)
