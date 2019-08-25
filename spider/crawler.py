from bs4 import BeautifulSoup
import pandas as pd
from spider.crawler_helpers import try_get_raw_html
import logging
from spider.config import DATA_SAVE_PATH, TARGET_URL


def get_review_data(div):
    review_title = div.select('h2.review-content__title')[0].text.replace('\n', '')
    review_text = div.select('p.review-content__text')[0].text.replace('\n', '')
    review_score = div.select('img')[0]['alt']
    review_id = div.select('a')[0]['href']
    return  review_title, review_text, review_score, review_id


if __name__ == '__main__':
    logging.info("Initiating crawl process on trustpilot website for company")
    crawled_data = []
    indexes = {}

    for i in range(250):
        logging.info('Processing page {0}'.format(i+1))
        raw_html = try_get_raw_html(page_number= i+1)

        if raw_html is None:
            break
        html = BeautifulSoup(raw_html, 'html.parser')
        for div in html.select('div.review-content'):
            review_title, review_text, review_score, review_id= get_review_data(div)
            crawled_data.append([review_id, review_title, review_text, review_score])

    data_df = pd.DataFrame(crawled_data, columns=['id','title','text','score'])
    data_df.to_csv(DATA_SAVE_PATH)
