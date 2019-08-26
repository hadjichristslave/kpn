import logging

import pandas as pd
from bs4 import BeautifulSoup

from spider.config import DATA_SAVE_PATH
from spider.crawler_helpers import try_get_raw_html


def get_review_data(div):
    title = div.select('h2.review-content__title')[0].text.replace('\n', '')
    text = div.select('p.review-content__text')[0].text.replace('\n', '')
    rating = div.select('img')[0]['alt']
    id = div.select('a')[0]['href']
    return  title, text, rating, id


if __name__ == '__main__':
    print("Initiating crawl process on TrustPilot website for company")
    crawled_data = []
    indexes = {}

    for page_number in range(250):
        print('Processing page {0}'.format(page_number + 1))
        raw_html = try_get_raw_html(page_number=page_number + 1)

        if raw_html is None:
            break
        html = BeautifulSoup(raw_html, 'html.parser')
        for div in html.select('div.review-content'):
            review_title, review_text, review_score, review_id= get_review_data(div)
            crawled_data.append([review_id, review_title, review_text, review_score])

    data_df = pd.DataFrame(crawled_data, columns=['id','title','text','score'])
    data_df.to_csv(DATA_SAVE_PATH)
