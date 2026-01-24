# locsum

Terminal tool for offline transcription and summarization of audio/video files.

## Hardware Requirements

The transcription can be run on a decent laptop without a GPU, but the summarization requires a powerful GPU in order to get good results. I could make it work on my [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) (8GB memory, 67 TOPS), but it didn't have enough memory to run LLM models with more than 8B parameters, so the summaries were not so good. I recently got a [ASUS Ascent GX10](https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/)  (128 GB memory, 1000 TFLOPS), which can run much bigger models and so can generate much better summaries. I'm currently using the 30B parameters `glm-4.7-flash` model with very satisfying results. The GX10 supposedly can run models up to 200B parameters, but for now there are not much models available on Ollama between 100B and 200B parameters.

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

## Whisper Installation

## Ollama Installation

## VPN Setup

The goal being to process our files locally, we might as well do it as much privately as possible. Here is how I installed and configured WireGuard VPN on my GX10.

First update your system with `sudo apt update && sudo apt upgrade`. If the kernel is updated during this step, a reboot is required before continuing.

- Install WireGuard: `sudo apt install wireguard`
- Download WireGuard configuration from my [Proton VPN](https://protonvpn.com/) account
- Copy the configuration file to `/etc/wireguard/protonvpn.conf` and `chown root:root` (with sudo)
- Test connection manually
  - Connect : `sudo wg-quick up protonvpn`
  - Check connection : `sudo wg`
  - Check IP address : `curl -4 ip.me`
  - Disconnect : `sudo wg-quick down protonvpn`
- Connect at boot : `sudo systemctl enable --now wg-quick@protonvpn.service`
- Reboot and check VPN connection / IP address

## Radio Deactivation

For a truly air-gapped system and to eliminate electromagnetic radiation, here is how I disabled the antennas:

- Disable Bluetooth:

```sh
sudo systemctl disable --now bluetooth
sudo rfkill block bluetooth
```

- Disable wifi:

```sh
sudo systemctl disable --now wpa_supplicant
sudo rfkill block wifi
nmcli radio wifi off
```

- Check:

```sh
sudo systemctl is-enabled bluetooth
sudo systemctl is-enabled wpa_supplicant
sudo rfkill list
nmcli general status
```

## License

Copyright (c) 2026 Monsieur Linux

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Thanks to the creators and contributors of the powerful [openai-whisper](https://github.com/openai/whisper) and [ollama](https://github.com/ollama/ollama-python) libraries for making this project possible.

