"""
Fetches the indexable content of GOV.UK pages using the search API, and prints
it as CSV to stdout.

Indexable content is format-dependent: it's whatever publishing applications
decide to pass to rummager when indexing the document.
"""
from requests_cache import CachedSession
import requests
import argparse
import csv
import sys
import time
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

HEADER = ['url', 'link', 'title', 'description', 'content', 'topics', 'organisations']

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
parser.add_argument('--environment', '-e', dest='root_url', default='https://www-origin.staging.publishing.service.gov.uk', help='the environment used to query the search API')
parser.add_argument('--skip', '-s', dest='skip', type=int, default=0, help='Number of input rows to skip. Can be used to resume a partially completed import')
parser.add_argument('--skip-redirects', '-r', dest='skip_redirects', action='store_true', help="Don't test URLs on GOV.UK to resolve redirected links.")
parser.add_argument('--wait-time', '-w', dest='wait_time', type=float, default=0.1, help='Time to wait between each link, to work around rate limiting.')
args = parser.parse_args()

session = CachedSession(cache_name='govuk_cache', backend='sqlite')
retries = Retry(total=5, backoff_factor=args.wait_time, status_forcelist=[ 429 ])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))


def test_base_path(original_base_path, args):
    """
    Given a base path, try and classify it as valid, redirected, or gone,
    so that we can fetch data even when the link has been redirected.

    If it can't be retrieved, return None, otherwise return the ultimate base path.

    We might include the same document multiple times in our analysis, but
    this should only happen for a small amount of links and we can strip
    out duplicates later.

    You can pass --skip_redirects flag on the commmand line, to skip this step,
    which will exclude redirects from the import.
    """
    if args.skip_redirects:
        return original_base_path

    # WARNING: some redirects are hardcoded to production URLs.
    # Both staging and production will rate limit us.
    response = session.head(args.root_url + original_base_path, allow_redirects=True)

    if 200 <= response.status_code < 300:
        return response.url.replace('https://www.gov.uk', '').replace(args.root_url, '')
    elif response.status_code == 429:
        response.raise_for_status()
    else:
        if response.status_code not in (410,):
            sys.stderr.write("Unexpected response {} for {}\n".format(response.status_code, original_base_path))
        return None


def request_search_result(link, search_url):
    """
    Fetch a single link from the search API
    """
    if link is None:
        return {}

    response = session.get(search_url, params=search_params_for_link(link))
    response.raise_for_status()
    results = response.json()['results']
    if len(results) != 1:
        sys.stderr.write('Unexpected number of results for {}: {}\n'.format(link, len(results)))
        return {}
    return results[0]


def search_params_for_link(link):
    """
    Parameters to pass to the search API
    """
    return {
        'filter_link': link,
        'debug': 'include_withdrawn',
        'fields[]': [
            'indexable_content',
            'title',
            'description',
            'expanded_organisations',
            'expanded_topics',
        ],
    }


def extract_base_path(input_row):
    """
    Extract the base path from a GOV.UK URL
    """
    try:
        link = input_row['url']
    except KeyError:
        raise KeyError('Input CSV should contain a column header "url"')

    return link, link.replace('https://www.gov.uk', '')


def format_topics(topics):
    """
    Encode tag titles as CSV
    """
    return '|'.join([topic.get('title', '') for topic in topics])


def format_value(text):
    """
    Format text. Replaces newlines and carriage returns with spaces so that
    each line of the output is one document.
    """
    return text.encode('utf8').replace('\n', ' ').replace('\r', ' ')


def format_result(url, link, search_result):
    """
    Build a row of the output CSV from the search result
    Each value is a utf8 bytestring.
    """
    return {
        'url': url,
        'link': link,
        'content': format_value(search_result.get('indexable_content', '')),
        'description': format_value(search_result.get('description', '')),
        'title': format_value(search_result.get('title', '')),
        'topics': format_topics(search_result.get('expanded_topics', [])),
        'organisations': format_topics(search_result.get('expanded_organisations', [])),
    }


def fetch_rows(links_input_file, args):
    """
    Iterate through the input rows and yield the output rows
    """
    root_url = args.root_url
    skip = args.skip

    for i, row in enumerate(csv.DictReader(links_input_file)):
        if i < skip:
            continue

        if i % 100 == 0 and i > 0:
            sys.stderr.write('Row {}\n'.format(i + 1))

        url, original_base_path = extract_base_path(row)
        real_base_path = test_base_path(original_base_path, args)
        search_url = root_url + '/api/search.json'
        search_result = request_search_result(link=real_base_path, search_url=search_url)

        yield format_result(url, real_base_path, search_result)


if __name__ == '__main__':
    output = csv.DictWriter(sys.stdout, fieldnames=HEADER)
    output.writeheader()

    for row in fetch_rows(links_input_file=args.input_file, args=args):
        output.writerow(row)
        sys.stdout.flush()
