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
