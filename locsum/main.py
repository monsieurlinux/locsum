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
import pymupdf
import re
import shutil
import subprocess
import sys
import time
import tomllib
import warnings
from datetime import datetime
from pathlib import Path

# Third-party library imports
import markdown_it
import ollama
from weasyprint import HTML

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

CONFIG = {}

BLACK   = '\033[30m'
RED     = '\033[31m'
GREEN   = '\033[32m'
YELLOW  = '\033[33m'
BLUE    = '\033[34m'
MAGENTA = '\033[35m'
CYAN    = '\033[36m'
WHITE   = '\033[37m'
RESET   = '\033[0m'

# Get a logger for this script
logger = logging.getLogger(__name__)


def main():
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
        load_config(args.reset_config)
    except FileNotFoundError as e:
        print(f'{RED}Error:{RESET} Failed to load configuration file: {e}')
        return

    if normalize_path(CONFIG['whisper_cpp']['cli_path']).exists():
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

    # Set Ollama model and prompt
    ollama_prompt = CONFIG['ollama']['prompt']
    if args.ollama_model:
        ollama_model = args.ollama_model
    elif args.tiny:
        ollama_model = CONFIG['ollama']['tiny_model']
    else:
        ollama_model = CONFIG['ollama']['model']

    # Check if Ollama model available
    if not is_model_available(ollama_model):
        # We could pull it automatically, but unlike with Whisper no progress
        # bar would be displayed.
        print(f'{RED}Error:{RESET} The {ollama_model} model is not available, please pull it with `ollama pull {ollama_model}`')
        return

    # Get Ollama model's context length
    ctx_len = get_context_length(ollama_model)
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
        truncated = truncate_to_terminal(filename, padding = processing)
        print(f'{processing}{BLUE}{truncated}{RESET}')
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
            if whisper_engine == 'cpp':
                transcript_text = transcribe_whisper_cpp(
                    filename, whisper_model, language)
            else:
                transcript_text = transcribe_whisper_std(
                    filename, whisper_model, language)
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
            if not summary_text:
                # We are starting with a 'md' file
                summary_text = read_file(filename)
            pdf_bytes = write_pdf(pdf_file, summary_text, 'regular.css')

            if not args.no_compact:
                # TODO: Move threshold to configuration file
                # Regenerate pdf with compact layout if last page very short
                last_page_len = get_last_page_len(pdf_bytes)
                i = 1
                
                while 0 < last_page_len < 1500:
                    if i <= 1:
                        logger.debug(f'Last page is short, compact PDF')
                    else:
                        logger.debug(f'Last page is still short, compact more')
                    pdf_bytes = write_pdf(pdf_file, summary_text, f'compact{i}.css')
                    last_page_len = get_last_page_len(pdf_bytes)
                    if i >= 3: break
                    i += 1

        exec_time = time.time() - start_time
        if exec_time > 5:
            print(f'File processed in {WHITE}{format_time(exec_time)}{RESET}')

    if num_files > 1:
        all_exec_time = time.time() - all_start_time
        print(f'All files processed in {GREEN}{format_time(all_exec_time)}{RESET}')


def get_last_page_len(pdf_bytes):
    last_page_len = -1

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        last_page_len = len(doc.load_page(len(doc) - 1).get_text())
        logger.debug(f'Last page contains {last_page_len} characters')

    return last_page_len


def transcribe_whisper_std(filename, model_name, language):
    # Transcribe with Whisper
    # Models are stored in ~/.cache/whisper/
    model = whisper.load_model(model_name)

    print(f'Transcribing with {YELLOW}{model_name}{RESET} model')
    start_time = time.time()
    result = model.transcribe(filename, language=language)
    exec_time = time.time() - start_time
    logger.debug(f'Done in {format_time(exec_time)}')

    if not result['text']:
        logger.error(f'{RED}Transcription failed{RESET}')

    return result['text']


def transcribe_whisper_cpp(filename, model_name, language):
    # Transcribe with whisper.cpp
    # Models are stored in whisper.cpp/models/
    cli_path = Path(CONFIG['whisper_cpp']['cli_path'])
    model_path = Path(CONFIG['whisper_cpp']['models_path']) / model_name
    cli_path = normalize_path(cli_path, must_exist=True)
    model_path = normalize_path(model_path, must_exist=True)
    threads = str(CONFIG['whisper_cpp']['threads'])
    processors = str(CONFIG['whisper_cpp']['processors'])

    # https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli
    cmd = [cli_path, '-m', model_path, '-f', filename,
           '-l', language, '-t', threads, '-p', processors,
           '--no-timestamps']

    print(f'Transcribing with {YELLOW}{model_name}{RESET} model')
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    exec_time = time.time() - start_time
    logger.debug(f'Done in {format_time(exec_time)}')

    if not result.stdout:
        logger.error(f'{RED}Transcription failed{RESET}')

    return result.stdout


"""
def transcribe_faster_whisper(filename, model_name, language):
    # Transcribe with faster-whisper
    # Models are stored in ~/.cache/huggingface/hub/
    from faster_whisper import WhisperModel

    # device="cuda"                # is it the default?
    # compute_type="float16"       # best tradeoff: fast + accurate
    # compute_type="int8_float16"  # even faster, slightly lower accuracy
    # try turbo model
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    print(f'Transcribing with {model_name} model on {model.device} device')
    #print(f'Transcribing with {YELLOW}{model_name}{RESET} model')

    start_time = time.time()
    segments, info = model.transcribe(filename, language=language, beam_size=5)
    segments = list(segments)

    # Strip each segment, skip empties, join with newlines
    text = "\n".join(seg.text.strip() for seg in segments if seg.text.strip())

    exec_time = time.time() - start_time
    logger.debug(f'Done in {format_time(exec_time)}')

    return text
"""


def test_model_speed(transcript_text, ollama_prompt):
    runs = 10
    times_q4km = []
    times_q8_0 = []
    times_bf16 = []

    for i in range(runs):
        model = 'glm-4.7-flash'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model, ollama_prompt)
        end = time.perf_counter()
        times_q4km.append(end - start)

        model = 'glm-4.7-flash:q8_0'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model, ollama_prompt)
        end = time.perf_counter()
        times_q8_0.append(end - start)

        model = 'glm-4.7-flash:bf16'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model, ollama_prompt)
        end = time.perf_counter()
        times_bf16.append(end - start)

        avg_q4km = sum(times_q4km) / len(times_q4km)
        avg_q8_0 = sum(times_q8_0) / len(times_q8_0)
        avg_bf16 = sum(times_bf16) / len(times_bf16)

        print(f"Average time for q4km: {avg_q4km} seconds")  # 69.1 sec
        print(f"Average time for q8_0: {avg_q8_0} seconds")  # 86.6 sec (+25%)
        print(f"Average time for bf16: {avg_bf16} seconds")  # 132.4 sec (+92%)


def summarize(transcript, model, prompt):
    # Summarize with Ollama
    print(f'Summarizing with {YELLOW}{model}{RESET} model')
    start_time = time.time()

    # Setup your input and the initial context
    # Initialize the conversation list
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant specializing in detailed summaries."
        }
    ]

    # First Request: Summarize the text
    # We send the system prompt + the text to summarize
    messages.append({"role": "user", "content": f"{prompt}\n\n{transcript}"})

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

    # Loop: Check length and request details if too short
    # TODO: Maybe replace 'if' by 'while', but put a limit on the number of iterations
    if ratio_pct < target_ratio:
        print(f"Summary is too short ({RED}{ratio_pct:.1f}%{RESET} ratio for {GREEN}{target_ratio}%{RESET} target), asking for more details")
        start_time = time.time()
        
        # Append a new user instruction
        # IMPORTANT: We also append the previous 'assistant' message 
        # (the current summary) so the model has context.
        messages.append({
            "role": "user", 
            #"content": "Your summary is too short. Could you tell me more about that in detail?"
            "content": "Could you tell me more about that in detail?"
        })
        
        # Get the new response
        response = ollama.chat(model=model, messages=messages)
        summary = response['message']['content']
        exec_time = time.time() - start_time
        ratio_pct = len(summary) / len(transcript) * 100
        logger.debug(f'Done in {format_time(exec_time)} ({ratio_pct:.1f}% ratio)')
        
        color = RED if ratio_pct < target_ratio else GREEN
        print(f"New summary has a {color}{ratio_pct:.1f}%{RESET} ratio")
        
        # Append this new response to history for the next iteration
        messages.append({"role": "assistant", "content": summary})

    return summary


def is_model_available(model_name: str) -> bool:
    # Fetch local models
    models = ollama.list()['models']

    # Extract just the names into a list
    names = [m['model'] for m in models]

    # Check for exact match or with 'latest' suffix
    return model_name in names or f'{model_name}:latest' in names


def get_context_length(model_name: str) -> int:
    try:
        modelinfo = ollama.show(model_name).get("modelinfo")

        if not isinstance(modelinfo, dict):
            logger.debug(f"'modelinfo' not found or not a dict for model '{model_name}'")
            return 0

        # Look for any key ending with '.context_length'
        for key, value in modelinfo.items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    logger.debug(f"Context length value for key '{key}' is not an integer: {value}")
                    continue

        logger.debug(f"No '.context_length' key found in modelinfo for '{model_name}'")
        return 0

    except Exception as e:
        logger.debug(f"Error fetching model info for '{model_name}': {e}")
        return 0


def write_pdf(pdf_file, md_content, css_file):
    # Parse markdown
    md = markdown_it.MarkdownIt()
    html_content = md.render(md_content)
    date = datetime.now().strftime('%Y-%m-%d')
    header = get_file_stem(pdf_file) + ' / ' + date

    # CSS styling
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
    #logger.debug(f'Wrote to {pdf_file}')
    return pdf_bytes


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def write_file(filename, content, mode='w'):
    with open(filename, mode) as file:
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


def add_suffix(filename, suffix = ''):
    p = Path(filename)
    return f'{p.parent}/{p.stem}{suffix}{p.suffix}'


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


def truncate_to_terminal(text, padding=''):
    width = shutil.get_terminal_size().columns - len(padding)

    # Make space for full-width unicode characters
    str_width = sum(2 if ord(c) > 127 else 1 for c in text)
    width -= (str_width - len(text))

    if len(text) <= width:
        return text
    else:
        ellipsis = "..."
        truncated = text[:width - len(ellipsis)]
        return truncated + ellipsis


def normalize_path(path, *, must_exist=False):
    p = Path(path)
    p = p.expanduser()  # Expand ~
    p = p.absolute()    # Convert to absolute

    if must_exist:
        p = p.resolve() # Resolve symlinks and validate existence (?)

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")

    return p


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
