# Standard library imports
import pymupdf
from datetime import datetime
from pathlib import Path

# Third-party library imports
import markdown_it
from weasyprint import HTML

# Local imports
from logger import logger
from utils import get_file_stem, read_file, write_file


def write_pdf(pdf_file, md_content, css_file):
    # Parse markdown
    md = markdown_it.MarkdownIt()
    html_content = md.render(md_content)
    date = datetime.now().strftime('%Y-%m-%d')
    header = get_file_stem(pdf_file) + ' / ' + date

    # CSS styling
    # TODO: Automatically get 'locsum' to avoid hardcoding
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    css = read_file(PROJECT_ROOT / 'locsum' / css_file)
    
    # HTML code
    html = """
    <html>
    <head>
        <style>
            @page {
                size: letter;
                
                @top-center {
                    content: " """ + header + """ ";
                    font-size: 6pt;
                }
                
                @bottom-center {
                    content: counter(page) " / " counter(pages);
                    font-size: 6pt;
                }
            }
        </style>
        <style>""" + css + """</style>
    </head>
    <body>
        """ + html_content + """
    </body>
    </html>
    """
    
    pdf_bytes = HTML(string=html).write_pdf()
    write_file(pdf_file, pdf_bytes, mode='wb')
    return pdf_bytes


def get_num_pages(pdf_bytes):
    num_pages = -1

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        num_pages = len(doc)
        logger.debug(f'PDF contains {num_pages} pages')

    return num_pages


def get_last_page_len(pdf_bytes):
    last_page_len = -1

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        last_page_len = len(doc.load_page(len(doc) - 1).get_text())
        logger.debug(f'Last page contains {last_page_len} characters')

    return last_page_len


