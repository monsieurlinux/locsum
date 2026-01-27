#!/usr/bin/env python3

"""
Copyright (c) 2026 Monsieur Linux

Licensed under the MIT License. See the LICENSE file for details.
"""

# Standard library imports
import argparse
import glob
import logging
#import os
#import shutil
import re
import sys
import time
#import tomllib
import warnings
from datetime import datetime
from pathlib import Path

# Third-party library imports
import markdown_it
import ollama
#import torch
from weasyprint import HTML
import whisper

# Add project root to sys.path so script can be called directly w/o 'python3 -m'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from locsum import __version__

# Configuration constants
DEFAULT_LANGUAGE = 'en'
WHISPER_MODEL_ENGLISH = 'base.en'
WHISPER_MODEL_MULTILINGUAL = 'turbo'
LLM_MODEL = 'glm-4.7-flash'
LLM_PROMPT = "Please provide a comprehensive, detailed, and structured breakdown of the following text. Do not just list events sequentially; instead, analyze the content and group it into distinct themes or categories. For each theme include a clear, bolded heading. Ensure the summary captures the full nuance of the speaker's opinions, including any criticisms, predictions, or advice offered. The tone should be informative and objective, accurately reflecting the source material. Here is the text:"

# Get a logger for this script
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filenames', nargs='+', metavar='FILE',
                        help='audio/video file to process')
    parser.add_argument('-l', '--language', default=DEFAULT_LANGUAGE,
                        help='set the language of the audio (default: en)')
    parser.add_argument('-t', '--tiny', action='store_true',
                        help='use tiny models for testing')
    parser.add_argument('-v', '--version', action='version', 
                        version=f'%(prog)s {__version__}')
    parser.add_argument('-w', '--filter-warnings', action='store_true',
                        help='suppress warnings from torch')
    args = parser.parse_args()

    if args.filter_warnings:
        # Suppress all CUDA-related warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

        # Or suppress all warnings from torch
        #warnings.filterwarnings("ignore", module="torch")

    if args.tiny:
        # TODO: Find a clean way to do this
        global WHISPER_MODEL_ENGLISH
        global WHISPER_MODEL_MULTILINGUAL
        global LLM_MODEL
        global LLM_PROMPT
        WHISPER_MODEL_ENGLISH = 'tiny.en'
        WHISPER_MODEL_MULTILINGUAL = 'tiny'
        LLM_MODEL = 'tinyllama'
        LLM_PROMPT = "Summarize:"

    all_start_time = time.time()
    filenames = []

    if sys.platform == "win32":
        # On Windows, expand glob patterns (e.g. *.mp4)
        for pattern in args.filenames:
            filenames.extend(glob.glob(pattern))
    else:
        # On Linux, use filenames as-is (no glob expansion needed)
        filenames = args.filenames

    for filename in filenames:
        print(f'Processing {filename}')
        start_time = time.time()
        extension = get_file_extension(filename)
        transcript_text = None
        summary_text = None
        next_step = 'txt'
        
        if extension == 'txt':
            # Processing a 'txt' file, so skip to summarization
            next_step = 'md'
        elif extension == 'md':
            # Processing a 'md' file, so skip to pdf generation
            next_step = 'pdf'

        if next_step == 'txt':
            # Assume audio file, attempt transcription
            txt_file = replace_extension(filename, 'txt')
            transcript_text = transcribe(filename, args.language)
            write_file(txt_file, transcript_text)
            next_step = 'md'

        if next_step == 'md':
            # Generate a summary from the transcription
            md_file = replace_extension(filename, 'md')
            if not transcript_text:
                # We are starting with a 'txt' file
                transcript_text = read_file(filename)
            summary_text = summarize(transcript_text)
            write_file(md_file, summary_text)
            next_step = 'pdf'

        if next_step == 'pdf':
            # Generate a pdf from the summary
            pdf_file = replace_extension(filename, 'pdf')
            #pdf_file = cleanup_filename(pdf_file)
            if not summary_text:
                # We are starting with a 'md' file
                summary_text = read_file(filename)
            write_pdf(pdf_file, summary_text)

        exec_time = time.time() - start_time
        print(f'File processed in {format_time(exec_time)}')

    all_exec_time = time.time() - all_start_time
    print(f'All files processed in {format_time(all_exec_time)}')


def transcribe(filename, language):
    # Transcribe with Whisper
    if language == 'en':
        whisper_model = WHISPER_MODEL_ENGLISH
    else:
        whisper_model = WHISPER_MODEL_MULTILINGUAL

    # It isn't necessary to import torch and explicitely load the model to CUDA.
    # Whisper library handles device detection automatically.
    model = whisper.load_model(whisper_model)
    print(f'Transcribing with {whisper_model} model on {model.device} device')
    start_time = time.time()
    result = model.transcribe(filename, language=language)
    exec_time = time.time() - start_time
    print(f'Done in {format_time(exec_time)}')

    return result['text']


def summarize(transcript):
    # Summarize with Ollama
    print(f'Summarizing with {LLM_MODEL} model')
    start_time = time.time()

    response = ollama.chat(model=LLM_MODEL, messages=[
      {
        'role': 'user',
        'content': f'{LLM_PROMPT} {transcript}',
      },
    ])

    exec_time = time.time() - start_time
    print(f'Done in {format_time(exec_time)}')
    summary = response['message']['content']

    return summary


def write_pdf(pdf_file, md_content):
    # Parse markdown
    md = markdown_it.MarkdownIt()
    html_content = md.render(md_content)
    date = datetime.now().strftime('%Y-%m-%d')
    header = get_file_stem(pdf_file) + ' / ' + date

    # CSS styling
    css = read_file(PROJECT_ROOT / 'locsum' / 'pdf.css')
    
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
    
    HTML(string=html).write_pdf(pdf_file)
    logger.debug(f'Wrote to {pdf_file}')


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def write_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
    logger.debug(f'Wrote to {filename}')


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    logger.debug(f'Read from {filename}')
    return content


def get_head_tail(s, head_len=40, tail_len=40, sep="..."):
    return (s[:head_len] + sep + s[-tail_len:])


def get_file_extension(filename):
    p = Path(filename)
    return p.suffix[1:]  # Remove the leading dot


def get_file_stem(filename):
    p = Path(filename)
    return p.stem


def replace_extension(filename, extension = ''):
    p = Path(filename)
    return f'{p.parent}/{p.stem}.{extension}'


def cleanup_filename(filename):
    p = Path(filename)
    stem = re.sub(r"[^a-zA-Z0-9 .,'_-]", '-', p.stem)
    return f'{p.parent}/{stem}{p.suffix}'


def setup_logging(level=logging.DEBUG):
    """Configure logging for this module"""
    # print() is for user consumption, logging is for developer consumption
    #logger.handlers.clear()  # Remove any existing handlers from your logger
    if not logger.handlers:  # Prevent duplicate handlers
        # TODO: Optionaly make the call to basicConfig if I need to
        handler = logging.StreamHandler()  # pass sys.stdout?
        handler.setLevel(level)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False  # Don't bubble up to root


if __name__ == '__main__':
    #setup_logging()  # Try this instead if messages are not displayed

    # Configure the root logger
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')

    # Set level for all existing loggers (notably from ttFont module)
    for name in logging.Logger.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Configure this script's logger
    logger.setLevel(logging.DEBUG)

    main()
