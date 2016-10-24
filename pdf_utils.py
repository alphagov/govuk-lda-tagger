import StringIO
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

def convert_pdf_to_text(pdf_link):
    resource_manager = PDFResourceManager()
    output = StringIO.StringIO()
    device = TextConverter(
        resource_manager,
        output,
        codec='utf-8',
    )
    interpreter = PDFPageInterpreter(resource_manager, device)
    file_handler = file(pdf_link, 'rb')
    pages = PDFPage.get_pages(file_handler)

    for page in pages:
        interpreter.process_page(page)

    return output.getvalue().strip('\n\t\r')
