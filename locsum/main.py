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
#import tomllib
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
    parser.add_argument('-s', '--skip-transcription', action='store_true',
                        help='skip transcription, use cached version instead')
    parser.add_argument('-S', '--simulate', action='store_true',
                        help='skip transcription and summarization')
    parser.add_argument('-t', '--tiny', action='store_true',
                        help='using tiny models for testing')
    parser.add_argument('-v', '--version', action='version', 
                        version=f'%(prog)s {__version__}')
    args = parser.parse_args()

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

    # Create a list containing all files from all patterns like '*.m4a',
    # because under Windows the terminal doesn't expand wildcard arguments.
    all_files = []
    for pattern in args.filenames:
        all_files.extend(glob.glob(pattern))

    for filename in all_files:
        print(f'Processing {filename}')
        transcript_file = replace_extension(filename, 'txt')
        summary_file = replace_extension(filename, 'md')
        summary_file = cleanup_filename(summary_file)
        pdf_file = replace_extension(filename, 'pdf')

        if args.skip_transcription:
            transcript_text = read_file(transcript_file)
        else:
            if args.simulate:
                transcript_text = 'simulated transcription text'
                logger.info(f'Simulating transcription')
            else:
                transcript_text = transcribe(filename, args.language)
                write_file(transcript_file, transcript_text)

        if args.simulate:
            summary_text = 'simulated summary text'
            logger.info(f'Simulating summarization')
        else:
            summary_text = summarize(transcript_text)
            write_file(summary_file, summary_text)
            md2pdf(summary_file, pdf_file)


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
    result = model.transcribe(filename, language=language)

    return result['text']


def summarize(transcript):
    # Summarize with Ollama
    print(f'Summarizing with {LLM_MODEL} model')

    response = ollama.chat(model=LLM_MODEL, messages=[
      {
        'role': 'user',
        'content': f'{LLM_PROMPT} {transcript}',
      },
    ])

    summary = response['message']['content']

    return summary


def md2pdf(md_file, pdf_file):
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Parse markdown
    md = markdown_it.MarkdownIt()
    html_content = md.render(md_content)

    # CSS styling
    css = """
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 16px;
        line-height: 1.3;
        margin: 0;
        padding: 0;
    }

    code {
        font-family: 'Courier New', Courier, monospace;
    }

    p {
        margin: 1.2em 0;  /* top and bottom */
    }

    ul, ol {
        margin: 0;
        padding: 0 0 0 1.5em;  /* left only */
    }

    li {
        margin: 0.4em 0;  /* top and bottom */
        padding: 0;
    }
    """

    # HTML code
    html = f"""
    <html>
    <head>
        <style>{css}</style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    HTML(string=html).write_pdf(pdf_file)
    logger.info(f'Wrote to {pdf_file}')


def write_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
    logger.info(f'Wrote to {filename}')


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    logger.info(f'Read from {filename}')
    return content


def get_head_tail(s, head_len=40, tail_len=40, sep="..."):
    return (s[:head_len] + sep + s[-tail_len:])


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
                        format='%(levelname)s - %(message)s')

    # Set level for all existing loggers (notably from ttFont module)
    for name in logging.Logger.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Configure this script's logger
    logger.setLevel(logging.DEBUG)

    main()
