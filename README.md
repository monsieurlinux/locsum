# locsum

Terminal tool for offline transcription and summarization of audio/video files.

## Dependencies

locsum requires the following external libraries:

* **[ollama](https://github.com/ollama/ollama-python)**: Used for text summarization.
* **[openai-whisper](https://github.com/openai/whisper)**: Used for audio transcription.

These libraries and their sub-dependencies will be installed automatically when you install locsum.

## Installation

It is recommended to install locsum within a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to avoid conflicts with system packages. Some Linux distributions enforce this. You can use `pipx` to handle the virtual environment automatically, or create one manually and use `pip`.

### Installation with `pipx`

`pipx` installs locsum in an isolated environment and makes it available globally.

**1. Install `pipx`:**

*   **Linux (Debian / Ubuntu / Mint):**
    
    ```bash
    sudo apt install pipx
    pipx ensurepath
    ```
*   **Linux (Other) / macOS:**
    
    ```bash
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    ```
*   **Windows:**
    
    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

You may need to close and restart your terminal for the PATH changes to take effect.

**2. Install locsum:**

```bash
pipx install locsum
```

### Installation with `pip`

If you prefer to manage the virtual environment manually, you can create and activate it by following this [tutorial](https://docs.python.org/3/tutorial/venv.html). Then install locsum:

```bash
pip install locsum
```

## Usage

### Basic Usage

```bash
locsum [arguments] FILE [FILE ...]
```

### Command-Line Arguments

None for now.

## Configuration

(Not yet implemented.)

When you run locsum for the first time, a `config.toml` file is automatically created. Its location depends on your operating system (typical paths are listed below):

*   **Linux:** `~/.config/locsum`
*   **macOS:** `~/Library/Preferences/locsum`
*   **Windows:** `C:/Users/YourUsername/AppData/Roaming/locsum`

You can edit this file to customize various settings. Common customizations include....

## License

Copyright (c) 2026 Monsieur Linux

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Thanks to the creators and contributors of the powerful [openai-whisper](https://github.com/openai/whisper) and [ollama](https://github.com/ollama/ollama-python) libraries for making this project possible.

