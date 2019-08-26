FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN pip install scikit-learn numpy scipy flask uwsgi nltk==3.2.4
RUN pip install wordcloud==1.5.0
RUN pip install matplotlib



ENV LISTEN_PORT 8080
EXPOSE 8080


#COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY . /app
COPY model_api/flask_run.py /app/main.py
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV PYTHONPATH="$PYTHONPATH:/app/data:/app/model_api:/app/sentiment_analyzer:/app/spider:/app"


