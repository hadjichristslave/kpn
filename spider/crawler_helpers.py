from contextlib import closing
from spider.config import TARGET_URL
from requests import get
from retrying import retry
import logging
import traceback


@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def simple_get(url):
    """
    A retrying policy is implemented as it is best practice to have retries in network related long running processes
    :param url: URL to retrieve
    :return:
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_valid_response(resp):
                return resp.content
            else:
                return None
    except Exception as e:
        raise Exception(e)


def is_valid_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and resp.url != TARGET_URL
            and content_type.find('html') > -1)


def try_get_raw_html(page_number: int):
    try:
        request_url = '{0}?page={1}'.format(TARGET_URL, page_number)
        return simple_get(request_url)
    except Exception as ex:
        logging.warning("Failed to parse with exception message {0} and stack_trace of {1}."
                        " proceeding with requests as normal".format(ex, traceback.format_exc()))
