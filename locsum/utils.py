# Standard library imports
import re
import shutil
from pathlib import Path


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def normalize_path(path, *, must_exist=False):
    p = Path(path)
    p = p.expanduser()  # Expand ~
    p = p.absolute()    # Convert to absolute

    if must_exist:
        p = p.resolve() # Resolve symlinks and validate existence (?)

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")

    return p


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
