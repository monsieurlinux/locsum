#!/usr/bin/env python3

"""
Copyright (c) 2026 Monsieur Linux

Licensed under the MIT License. See the LICENSE file for details.
"""

# Standard library imports
import argparse
import glob
import logging
import os
import re
import shutil
import sys
import time
import tomllib
import warnings
from datetime import datetime
from pathlib import Path

# Third-party library imports
import markdown_it
import ollama
import torch
from weasyprint import HTML
import whisper

# Add project root to sys.path so script can be called directly w/o 'python3 -m'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from locsum import __version__

CONFIG = {}

# Get a logger for this script
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('filenames', nargs='*', metavar='FILE',
                        help='file to process (audio/video, .txt or .md format)')
    parser.add_argument('-c', '--check-cuda', action='store_true',
                        help='check if CUDA is available')
    parser.add_argument('-l', '--language', metavar='LANG',
                        help='set the language of the audio')
    parser.add_argument('-o', '--ollama-model', metavar='MODEL',
                        help='set the Ollama model for summarization')
    parser.add_argument('-r', '--reset-config', action='store_true',
                        help='reset configuration file to default')
    parser.add_argument('-t', '--transcribe-only', action='store_true',
                        help="transcribe only, don't generate a summary")
    parser.add_argument('-T', '--tiny', action='store_true',
                        help='use tiny Whisper and Ollama models for testing')
    parser.add_argument('-v', '--version', action='version', 
                        version=f'%(prog)s {__version__}')
    parser.add_argument('-w', '--whisper-model', metavar='MODEL',
                        help='set the Whisper model for transcription')
    parser.add_argument('-W', '--filter-warnings', action='store_true',
                        help='suppress warnings from PyTorch')
    args = parser.parse_args()

    try:
        load_config(args.reset_config)
    except FileNotFoundError as e:
        print(f'Error: Failed to load configuration file: {e}')
        return

    if args.filter_warnings:
        # Suppress all CUDA-related warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

        # Or suppress all warnings from torch
        #warnings.filterwarnings("ignore", module="torch")

    if args.check_cuda:
        # Check if CUDA is available
        print(f'PyTorch {torch.__version__}')
        if torch.cuda.is_available():
            print(f'CUDA {torch.version.cuda} is available')
        else:
            print('CUDA is NOT available')
        return

    if not args.filenames:
        # Check if files have been provided
        print('Error: The following arguments are required: FILE')
        return

    # Get default configuration
    whisper_language = CONFIG['whisper']['language']
    whisper_model = CONFIG['whisper']['model_multilang']
    ollama_model = CONFIG['ollama']['model']
    ollama_prompt = CONFIG['ollama']['prompt']

    # Set language
    if args.language:
        whisper_language = args.language

    # Set Whisper model
    if args.whisper_model:
        whisper_model = args.whisper_model
    elif args.tiny:
        whisper_model = CONFIG['whisper']['tiny']['model']
    elif whisper_language == 'en':
        whisper_model = CONFIG['whisper']['model_english']

    # Set Ollama model and prompt
    if args.ollama_model:
        ollama_model = args.ollama_model
    elif args.tiny:
        ollama_model = CONFIG['ollama']['tiny']['model']
        ollama_prompt = CONFIG['ollama']['tiny']['prompt']

    # Check if Ollama model available
    if not is_model_available(ollama_model):
        # We could pull it automatically, but unlike with Whisper no progress
        # bar would be displayed.
        print(f'Error: The {ollama_model} model is not available, please pull it with `ollama pull {ollama_model}`')
        return

    all_start_time = time.time()
    filenames = []
    num_files = 0

    if sys.platform == "win32":
        # On Windows, expand glob patterns (e.g. *.mp4)
        for pattern in args.filenames:
            filenames.extend(glob.glob(pattern))
    else:
        # On Linux, use filenames as-is (no glob expansion needed)
        filenames = args.filenames

    for filename in filenames:
        if not Path(filename).is_file():
            print(f'Skipping {filename} (not a file)')
            continue

        print(f'Processing {filename}')
        num_files += 1
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
            transcript_text = transcribe(filename, whisper_model, whisper_language)
            write_file(txt_file, transcript_text)
            if args.transcribe_only:
                next_step = 'none'
            else:
                next_step = 'md'

        if next_step == 'md':
            # Generate a summary from the transcription
            md_file = replace_extension(filename, 'md')
            if not transcript_text:
                # We are starting with a 'txt' file
                transcript_text = read_file(filename)
            summary_text = summarize(transcript_text, ollama_model, ollama_prompt)
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

    if num_files > 1:
        all_exec_time = time.time() - all_start_time
        print(f'All files processed in {format_time(all_exec_time)}')


def transcribe(filename, model_name, language):
    # Transcribe with Whisper. It isn't necessary to explicitely load the model
    # to CUDA. Whisper handles device detection automatically.
    model = whisper.load_model(model_name)
    #print(f'Transcribing with {model_name} model on {model.device} device')
    print(f'Transcribing with {model_name} model')
    start_time = time.time()
    result = model.transcribe(filename, language=language)
    exec_time = time.time() - start_time
    logger.debug(f'Done in {format_time(exec_time)}')

    return result['text']


"""
3 pages pdf ~ 9k chars, so ~ 3k chars per page

Make sure your summary is at least x% the size of the transcript.

venice:
Could you tell me more about that in detail?

qwen3-coder:
Please provide a comprehensive summary that is at least 1500 characters long.
Include all major points, key details, and important information from the transcript.
The summary should be detailed and well-structured.

Please provide an e xtremely detailed summary that is at least 1500 characters long.
You must include ALL key points, important details, and significant information from the transcript.
The summary should be comprehensive, well-structured, and detailed.
"""

def summarize(transcript, model, prompt):
    # Summarize with Ollama
    print(f'Summarizing with {model} model')
    start_time = time.time()

    # 1. Setup your input and the initial context
    # Initialize the conversation list
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant specializing in detailed summaries."
        }
    ]

    # 2. First Request: Summarize the text
    # We send the system prompt + the text to summarize
    #messages.append({"role": "user", "content": f"Summarize this text: {transcript}"})
    messages.append({"role": "user", "content": f"{prompt} {transcript}"})

    response = ollama.chat(model=model, messages=messages)
    summary = response['message']['content']
    exec_time = time.time() - start_time
    ratio_pct = len(summary) / len(transcript) * 100
    logger.debug(f'Done in {format_time(exec_time)} ({ratio_pct:.1f}% ratio)')

    # Add the first response to history so the model remembers what it wrote
    messages.append({"role": "assistant", "content": summary})
    
    # TODO: Make target ratios configurable
    transcript_size = len(transcript)
    
    if transcript_size < 25000:
        target_ratio = 10
    elif transcript_size < 50000:
        target_ratio = 6
    else:
        target_ratio = 4

    # 3. Loop: Check length and request details if too short
    # TODO: Maybe replace 'if' by 'while', but put a limit on the number of iterations
    if ratio_pct < target_ratio:
        print(f"Summary is too short ({ratio_pct:.1f}% ratio for {target_ratio}% target), asking for more details")
        start_time = time.time()
        
        # Append a new user instruction
        # IMPORTANT: We also append the previous 'assistant' message 
        # (the current summary) so the model has context.
        messages.append({
            "role": "user", 
            "content": "The summary I just provided was too short. Please expand on the key points and provide more detail without changing the original meaning."
        })
        
        # Get the new response
        response = ollama.chat(model=model, messages=messages)
        summary = response['message']['content']
        exec_time = time.time() - start_time
        ratio_pct = len(summary) / len(transcript) * 100
        logger.debug(f'Done in {format_time(exec_time)} ({ratio_pct:.1f}% ratio)')
        
        # Append this new response to history for the next iteration
        messages.append({"role": "assistant", "content": summary})

    return summary


def is_model_available(model: str) -> bool:
    # Fetch local models
    models = ollama.list()['models']

    # Extract just the names into a list
    names = [m['model'] for m in models]

    # Check for exact match or with 'latest' suffix
    return model in names or f'{model}:latest' in names
    

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
    #logger.debug(f'Wrote to {pdf_file}')


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def write_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
    #logger.debug(f'Wrote to {filename}')


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    #logger.debug(f'Read from {filename}')
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


def load_config(reset_config = False):
    global CONFIG

    app_name = 'locsum'
    config_file = 'config.toml'

    config_dir = get_config_dir(app_name)
    user_config_file = config_dir / config_file
    default_config_file = PROJECT_ROOT / app_name / config_file

    if not user_config_file.exists() or reset_config:
        if default_config_file.exists():
            shutil.copy2(default_config_file, user_config_file)
            logger.debug(f'Config initialized at {user_config_file}')
        else:
            raise FileNotFoundError(f'Default config missing at {default_config_file}')
    else:
        logger.debug(f'Found config file at {user_config_file}')

    with open(user_config_file, 'rb') as f:
        CONFIG = tomllib.load(f)


def get_config_dir(app_name):
    if sys.platform == "win32":
        # Windows: Use %APPDATA% (%USERPROFILE%\AppData\Roaming)
        config_dir = Path(os.environ.get("APPDATA", "")) / app_name
    elif sys.platform == "darwin":
        # macOS: Use ~/Library/Preferences
        config_dir = Path.home() / "Library" / "Preferences" / app_name
    else:
        # Linux and other Unix-like: Use ~/.config or XDG_CONFIG_HOME if set
        config_home = os.environ.get("XDG_CONFIG_HOME", "")
        if config_home:
            config_dir = Path(config_home) / app_name
        else:
            config_dir = Path.home() / ".config" / app_name
    
    # Create the directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    
    return config_dir


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
