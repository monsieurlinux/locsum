# Standard library imports
import subprocess
import time
from pathlib import Path

# Third-party library imports
try:
    import whisper
except ImportError:
    pass

# Local imports
from .colors import YELLOW, RED, RESET
from .logger import logger
from .utils import format_time, normalize_path


def transcribe_whisper_std(filename, model_name, language, config):
    # Transcribe with Whisper
    # Models are stored in ~/.cache/whisper/
    CONFIG = config
    model = whisper.load_model(model_name)

    print(f'Transcribing with {YELLOW}{model_name}{RESET} model')
    start_time = time.time()
    result = model.transcribe(filename, language=language)
    exec_time = time.time() - start_time
    logger.debug(f'Done in {format_time(exec_time)}')

    if not result['text']:
        logger.error(f'{RED}Transcription failed{RESET}')

    return result['text']


def transcribe_whisper_cpp(filename, model_name, language, config):
    # Transcribe with whisper.cpp
    # Models are stored in whisper.cpp/models/
    CONFIG = config
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


