#!/usr/bin/env python3

"""
Copyright (c) 2026 Monsieur Linux

Licensed under the MIT License. See the LICENSE file for details.
"""

# Standard library imports
import argparse
import glob
import logging
import sys
import time
import warnings
from pathlib import Path

# Third-party library imports
try:
    import torch
    import whisper
    HAS_WHISPER_STD = True
except ImportError:
    HAS_WHISPER_STD = False

# Add project root to sys.path so script can be called directly w/o 'python3 -m'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from locsum import __version__
import config
import pdfgenerator
import utils
import summarizer
import transcriber
from colors import BLUE, WHITE, GREEN, YELLOW, RED, RESET
from logger import logger
from utils import format_time, read_file, write_file

CONFIG = {}


def main():
    global CONFIG

    parser = argparse.ArgumentParser()

    parser.add_argument('filenames', nargs='*', metavar='FILE',
                        help='file to process (audio/video, .txt or .md format)')
    if HAS_WHISPER_STD:
        parser.add_argument('-c', '--check-cuda', action='store_true',
                            help='check if CUDA is available')
    parser.add_argument('-l', '--language', metavar='LANG',
                        help='set the language of the audio')
    parser.add_argument('-n', '--no-colors', action='store_true',
                        help="disable color output")
    parser.add_argument('-N', '--no-compact', action='store_true',
                        help="disable PDF compaction")
    parser.add_argument('-o', '--ollama-model', metavar='MODEL',
                        help='set the Ollama model for summarization')
    if HAS_WHISPER_STD:
        parser.add_argument('-O', '--openai-whisper', action='store_true',
                            default=argparse.SUPPRESS,
                            help="use OpenAI's Whisper even "
                                 "if Whisper.cpp is available")
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
    if HAS_WHISPER_STD:
        parser.add_argument('-W', '--filter-warnings', action='store_true',
                            help='suppress warnings from PyTorch')
    args = parser.parse_args()

    if args.no_colors:
        global BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''

    try:
        config.load_config(args.reset_config)
        CONFIG = config.CONFIG
    except FileNotFoundError as e:
        print(f'{RED}Error:{RESET} Failed to load configuration file: {e}')
        return

    if utils.normalize_path(CONFIG['whisper_cpp']['cli_path']).exists():
        HAS_WHISPER_CPP = True
    else:
        HAS_WHISPER_CPP = False

    if HAS_WHISPER_CPP and not hasattr(args, 'openai_whisper'):
        logger.debug("Using Whisper.cpp for transcription")
        whisper_engine = 'cpp'
    elif HAS_WHISPER_STD:
        logger.debug("Using OpenAI's Whisper for transcription")
        whisper_engine = 'std'
    else:
        print(f"{RED}Error:{RESET} No transcription engine is available, please install either OpenAI's Whisper or Whisper.cpp")
        return

    if whisper_engine == 'std' and args.filter_warnings:
        # Suppress all CUDA-related warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

        # Or suppress all warnings from torch
        #warnings.filterwarnings("ignore", module="torch")

    if whisper_engine == 'std' and args.check_cuda:
        # Check if CUDA is available
        print(f'PyTorch {torch.__version__}')
        if torch.cuda.is_available():
            print(f'CUDA {torch.version.cuda} is available')
        else:
            print('CUDA is NOT available')
        return

    if not args.filenames:
        # Check if files have been provided
        print(f'{RED}Error:{RESET} The following arguments are required: FILE')
        return

    # Set language
    if args.language:
        language = args.language
    else:
        language = CONFIG['audio']['language']

    # Set Whisper model
    if args.whisper_model:
        whisper_model = args.whisper_model
    elif whisper_engine == 'std':
        if args.tiny:
            whisper_model = CONFIG['whisper']['tiny_model']
        else:
            whisper_model = CONFIG['whisper']['model']
    elif whisper_engine == 'cpp':
        if args.tiny:
            whisper_model = CONFIG['whisper_cpp']['tiny_model']
        else:
            whisper_model = CONFIG['whisper_cpp']['model']

    # Set Ollama model
    if args.ollama_model:
        ollama_model = args.ollama_model
    elif args.tiny:
        ollama_model = CONFIG['ollama']['tiny_model']
    else:
        ollama_model = CONFIG['ollama']['model']

    # Check if Ollama model available
    if not summarizer.is_model_available(ollama_model):
        # We could pull it automatically, but unlike with Whisper no progress
        # bar would be displayed.
        print(f'{RED}Error:{RESET} The {ollama_model} model is not available, please pull it with `ollama pull {ollama_model}`')
        return

    # Get Ollama model's context length
    ctx_len = summarizer.get_context_length(ollama_model)
    if ctx_len > 0:
        logger.debug(f"Context length for {ollama_model} model: {ctx_len} tokens")
    else:
        print(f"{YELLOW}Warning:{RESET} Could not determine context length for {ollama_model} model")

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

        processing = 'Processing '
        truncated = utils.truncate_to_terminal(filename, padding = processing)
        print(f'{processing}{BLUE}{truncated}{RESET}')
        num_files += 1
        start_time = time.time()
        extension = utils.get_file_extension(filename)
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
            txt_file = utils.replace_extension(filename, 'txt')
            #transcript_text = transcribe(filename, whisper_engine,
            #                             whisper_model, language)
            if whisper_engine == 'cpp':
                transcript_text = transcriber.transcribe_whisper_cpp(
                    filename, whisper_model, language, CONFIG)
            else:
                transcript_text = transcriber.transcribe_whisper_std(
                    filename, whisper_model, language, CONFIG)
            write_file(txt_file, transcript_text)
            if args.transcribe_only:
                next_step = 'none'
            else:
                next_step = 'md'

        if next_step == 'md':
            # Generate a summary from the transcription
            md_file = utils.replace_extension(filename, 'md')
            if not transcript_text:
                # We are starting with a 'txt' file
                transcript_text = read_file(filename)
            summary_text = summarizer.summarize(transcript_text, ollama_model, CONFIG)
            write_file(md_file, summary_text)
            next_step = 'pdf'

        if next_step == 'pdf':
            # Generate a pdf from the summary
            pdf_file = utils.replace_extension(filename, 'pdf')
            if not summary_text:
                # We are starting with a 'md' file
                summary_text = read_file(filename)
            pdf_bytes = pdfgenerator.write_pdf(pdf_file, summary_text, 'regular.css')

            if not args.no_compact:
                # Regenerate pdf with compact layout if last page very short
                num_pages = pdfgenerator.get_num_pages(pdf_bytes)
                last_page_len = pdfgenerator.get_last_page_len(pdf_bytes)
                i = 1
                
                while num_pages > 1 and 0 < last_page_len < CONFIG['pdf']['short_page_threshold']:
                    if i <= 1:
                        logger.debug(f'Last page is short, compact PDF')
                    else:
                        logger.debug(f'Last page is still short, compact more')
                    pdf_bytes = pdfgenerator.write_pdf(pdf_file, summary_text, f'compact{i}.css')
                    num_pages = pdfgenerator.get_num_pages(pdf_bytes)
                    last_page_len = pdfgenerator.get_last_page_len(pdf_bytes)
                    if i >= 3: break
                    i += 1

        exec_time = time.time() - start_time
        if exec_time > 5:
            print(f'File processed in {WHITE}{format_time(exec_time)}{RESET}')

    if num_files > 1:
        all_exec_time = time.time() - all_start_time
        print(f'All files processed in {GREEN}{format_time(all_exec_time)}{RESET}')


"""
def transcribe(audio_path: str, engine: str, whisper_model: str, language: str):
    transcriber = create_transcriber(
        engine=engine,
        model_name=whisper_model,
        language=language
    )
    try:
        text = transcriber.transcribe(audio_path)
        print(f"Result: {text[:200]}...")
        return text
    except Exception as e:
        print(f"Transcription failed: {e}")
"""


def test_model_speed(transcript_text):
    runs = 10
    times_q4km = []
    times_q8_0 = []
    times_bf16 = []

    for i in range(runs):
        model = 'glm-4.7-flash'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model)
        end = time.perf_counter()
        times_q4km.append(end - start)

        model = 'glm-4.7-flash:q8_0'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model)
        end = time.perf_counter()
        times_q8_0.append(end - start)

        model = 'glm-4.7-flash:bf16'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model)
        end = time.perf_counter()
        times_bf16.append(end - start)

        avg_q4km = sum(times_q4km) / len(times_q4km)
        avg_q8_0 = sum(times_q8_0) / len(times_q8_0)
        avg_bf16 = sum(times_bf16) / len(times_bf16)

        print(f"Average time for q4km: {avg_q4km} seconds")  # 69.1 sec
        print(f"Average time for q8_0: {avg_q8_0} seconds")  # 86.6 sec (+25%)
        print(f"Average time for bf16: {avg_bf16} seconds")  # 132.4 sec (+92%)


if __name__ == '__main__':
    #setup_logging()  # Try this logger.py function if messages not displayed

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
