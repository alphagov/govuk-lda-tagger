from __future__ import print_function
import os
import sys
import ipdb
import uuid
import urllib
import StringIO
from time import sleep
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

def pdf_link_to_text(pdf_attachment):
    """
    Given a PDF URL, this function downloads it to a local file and extracts the
    text.
    """
    print("Downloading " + str(pdf_attachment))
    filename = download_pdf_file(pdf_attachment)
    print("Extracting text from " + str(filename))
    pdf_text = convert_pdf_to_text(filename)
    os.remove(filename)
    return pdf_text


def download_pdf_file(download_url):
    """
    Given a PDF URL, this function download the file to a local file.
    """
    web_file = urllib.urlopen(download_url)
    filename = "/tmp/" + str(uuid.uuid4()) + ".pdf"
    local_file = open(filename, 'w')
    local_file.write(web_file.read())
    web_file.close()
    local_file.close()
    return filename


def convert_pdf_to_text(pdf_path):
    """
    Given a path to a local PDF file, this function extracts text from it.
    """
    process_id = os.getpid()
    resource_manager = PDFResourceManager()
    output = StringIO.StringIO()
    laparams = LAParams(detect_vertical=True)
    device = TextConverter(
        resource_manager,
        output,
        codec='utf-8',
        laparams=laparams
    )
    interpreter = PDFPageInterpreter(resource_manager, device)
    file_handler = file(pdf_path, 'rb')
    pages = PDFPage.get_pages(file_handler)

    for idx, page in enumerate(pages):
        print("Page " + str(idx + 1), end='\r')
        sys.stdout.flush()
        interpreter.process_page(page)
    print()

    data = output.getvalue()
    data = data.replace('\n', ' ')
    data = data.replace('\t', ' ')
    data = data.replace('\r', ' ')
    data = data.replace('\x0c', ' ')

    return data
