import csv
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

def save_base_paths():
    base_paths_file = open('output/base_paths_file.csv', 'w')
    base_paths_file.write("\n".join(load_base_paths()))
    base_paths_file.close()
