# Rebel report
This repository introduces the solution code on the scraping and sentiment analysis pipeline based on reviews from Truspilot website.

The pipeline is split into three basic modules

  - Spider: Spider module scrapes and stores raw review data on the data/training_data folder
  - Sentiment analyzer: Processes the data into a valid format, and trains a plethora of classifiers. The best classifier is stored on data/models folder
  - Model API: Implements a scalable API that serves model predictions

# How to run the pipeline

The repository has a ready trained model, in order to re-scrape the site and retrain using newly scraped data, you need to run the following commands in the root of the application.
If you want to skip the retrain, you can go straight to the docker commands and run them from the root of your directory

### How to run the Retrain pipeline

Install the requirements.txt and run the following commands.

```sh
python3 -m venv envv
source envv/bin/activate
pip install -r requirements.txt
python3 -m spider.run
# Newly scraped repots should now reside on raw_dataset.csv
python3 -m sentiment_analyzer.run
```

### Docker installation

Once the program is run, you should now have a new best model and vectorizer on data/models folder
To now run the API you need to first install docker if you haven't done so already
Docker installation instructions can be found [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
Once you have a docker server up-and-running run the following commands
```

docker build -t sentiment:latest .
docker run -p 80:8000 --restart="always" sentiment:latest
```


Once the docker is live, you can query the API to get the predicted sentiment for different review inputs.
The sentiment prediction ranges from 1-5 with 1 being very negative sentiment and 5 being a very positive review sentiment
Some examples of API inputs can be found bellow
```sh
 curl 172.17.0.2:8000/predict_sentiment -d '{"title":"Stay Away!!", "text":"The worst experience fo my life!!"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8000/predict_sentiment -d '{"title":"Bad, should be better", "text":"needs a lot of improvement"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8000/predict_sentiment -d '{"title":"Nothing to brag about", "text":"Average services"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8000/predict_sentiment -d '{"title":"Nice", "text":"Could respond faster though"}' -H "Content-Type: application/json"
 curl 172.17.0.2:8000/predict_sentiment -d '{"title":"Best service ever!!", "text":"Nothing to add, you guys are the best!!"}' -H "Content-Type: application/json"
```

Different model configurations can be run by using the config.py file in sentiment_analyzer module.
Please keep in mind that the sentiment_analyzer needs to be re-run and the docker needs to be rebuilt afterwards in order for the model changes to take place.

Finally, in order to attain the accuracy presented in the report, svd needs to be run with 2000 components.
This leads to a very large model (~850MΒ) , so for the sake of presentation, the repo is left with a smaller model and no svd loaded when flask run is start


