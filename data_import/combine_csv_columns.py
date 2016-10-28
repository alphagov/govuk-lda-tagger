import csv
import sys

csv.field_size_limit(sys.maxsize)


def format_value(value):
    """
    When scraping indexable content from the search API, we joined
    organisation and topic titles with pipes. Since we are combining
    all the columns together here we need to make sure these get treated as
    separate words.
    """
    return value.replace('|', ' ')


if __name__ == '__main__':
    reader = csv.DictReader(sys.stdin)
    writer = csv.DictWriter(sys.stdout, ('url', 'words'))
    wordy_columns = set('title,description,content,topics,organisations,pdfdata'.split(','))
    writer.writeheader()

    for row in reader:
        words = ' '.join([format_value(value) for key, value in row.iteritems() if key in wordy_columns])
        writer.writerow(dict(url=row['url'], words=words))
