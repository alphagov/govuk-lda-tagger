import os
import csv
import ipdb
import json
import requests
import pdf_utils
import requests_cache
from urlparse import urlparse
from BeautifulSoup import BeautifulSoup


def fetch_education_urls(input_file):
    """
    Given a local input file, parse it and return a list of GOV.UK URLs.
    """
    documents = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        # skip headers
        next(reader, None)
        documents = list(reader)

    return [document[0] for document in documents]


def fetch_links_from_html(html_doc):
    """
    Given a blob of HTML, this function returns a list of PDF links
    """
    soup = BeautifulSoup(html_doc)
    pdf_attachments = []

    for link in soup.findAll('a'):
        value = link.get('href')
        if "http" in value:
            pdf_url = value
        else:
            pdf_url = "https://www.gov.uk" + value
        pdf_attachments.append(pdf_url)

    return pdf_attachments


def valid_pdf_attachment(pdf_attachment):
    """
    Given a string representing an attachment link, this function asserts if the
    link is a valid PDF link.
    """
    invalid_urls = ["https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/522875/BIS-16-22-evaluation-of-24_-advanced-learning-loans-an-assessment-of-the-first-year.pdf"]
    pdf_extension = ".pdf" in pdf_attachment
    not_mailto = "mailto" not in pdf_attachment
    not_invalid = pdf_attachment not in invalid_urls

    return pdf_extension and not_mailto and not_invalid


def fetch_text_from_pdf_attachments(url):
    """
    Given a GOV.UK URL, this function fetches the page from the content store,
    parses the PDF attachments (if any), and extracts the text from those
    attachments. It then returns the full text of the PDF attachments.
    """

    content_store_url = url.replace(
        "https://www.gov.uk/",
        "https://www-origin.staging.publishing.service.gov.uk/api/content/")

    r = requests.get(content_store_url)
    json_data = r.json()
    html_documents = json_data['details']['documents']
    pdf_attachments = []

    for html_doc in html_documents:
        pdf_attachments = fetch_links_from_html(html_doc)

    pdf_attachments = set(pdf_attachments)
    pdf_contents = []

    for pdf_attachment in pdf_attachments:
        if valid_pdf_attachment(pdf_attachment):
            pdf_text = pdf_utils.pdf_link_to_text(pdf_attachment)
            pdf_contents.append(pdf_text)

    all_pdf_content = str.join(" ", pdf_contents)

    print "     Success!"
    return all_pdf_content


if __name__ == "__main__":
    results = []
    requests_cache.install_cache()

    csvfile = open('output/urls_with_pdf_content.csv', 'wb')
    content_writer = csv.writer(csvfile, delimiter=',')

    urls = fetch_education_urls('input/all_audits_for_education.csv')
    for idx, url in enumerate(urls):
        try:
            print("===> Processing URL #" + str(idx) + " - " + url)
            data = fetch_text_from_pdf_attachments(url)
            content_writer.writerow([url, data])
        except ValueError as e:
            print e
            content_writer.writerow([url, ''])
        except KeyError as e:
            print "    => Document does not have attachments, skipping..."
            content_writer.writerow([url, ''])
