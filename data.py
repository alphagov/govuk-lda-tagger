import csv
import json
import urllib2
import urlparse

def load_links():
    with open('input/early-years-audit-all-content.csv', 'r') as f:
        reader = csv.reader(f)
        # skip headers
        next(reader, None)
        documents = list(reader)

    return [document[1] for document in documents if document[1] != '']

def load_base_paths():
    return [urlparse.urlparse(url).path for url in load_links()]

def dev_urls():
    return ["http://www.dev.gov.uk" + base_path + '?skip_slimmer=1' for base_path in load_base_paths()]

def save_base_paths():
    base_paths_file = open('output/base_paths_file.csv', 'w')
    base_paths_file.write("\n".join(load_base_paths()))
    base_paths_file.close()

def download_early_years_content():
    content = {}
    for base_path in load_base_paths():
        print "Fetching content for " + base_path
        url = 'https://www.gov.uk/api/search.json?filter_link={}&fields=indexable_content'.format(base_path)
        response = urllib2.urlopen(url)
        json_string = response.read()
        data = json.loads(json_string)
        results = data['results']
        if len(results) > 0:
            result = results[0]
            if 'indexable_content' in result:
                content[base_path] = result['indexable_content']

    return content
