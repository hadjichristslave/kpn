# Rebel report
This repository introduces the solution code on the scraping and sentiment analysis pipeline on the Truspilot website.

The pipeline is split into three basic modules

  - Spider: Spider module scrapes and stores raw review data on the data/training_data folder
  - Sentiment analyzer: Processses the data into a valid format, and trains a plethora of classifiers. The best classifier is stored on data/models folder
  - Model API: Implements a scalable API that serves model predictions

# How to run the pipline

The repository has a ready trained model, in order to re-scrape the site and retrain using newly scraped data, you need to run the following commands in the root of the application:

### Installation

Install the requirements.txt and run the following commands.

```sh
python3 -m venv envv
source envv/bin/activate
pip install -r requirements.txt
python3 -m spider.crawler
# Newly scraped repots should now reside on raw_dataset.csv
python3 -m sentiment_analyzer.main
```

Once the program is run, you should now have a new best model and vectorizer on data/models folder
To now run the API you need to first install docker if you haven't done so already
Docker installation instructions can be found [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
Once you have a docker server up-and-running run the following commands
```

docker build -t sentiment:latest .
docker run 80:8000 --restart="always" sentiment:latest
```


Once the docker is live, you can query the API to get the predicted sentiment for different review inputs.
Some examples of API inputs can be found bellow
```sh
 curl 172.17.0.2:8080/predict_sentiment -d '{"title":"Stay Away!!", "text":"The worst experience fo my life!!"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8080/predict_sentiment -d '{"title":"Bad, should be better", "text":"needs a lot of improvement"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8080/predict_sentiment -d '{"title":"Nothing to brag about", "text":"Average services"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8080/predict_sentiment -d '{"title":"Nice", "text":"Could respond faster though"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8080/predict_sentiment -d '{"title":"Best service ever!!", "text":"Nothing to add, you guys are the best!!"}' -H "Content-Type: application/json"
```



