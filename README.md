# Locsum

[![PyPI][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]

Terminal tool for batch offline transcription and summarization of audio/video files.

## Hardware Requirements

Transcription can run on a CPU without a GPU, but high-quality summarization requires significant GPU resources. I initially used an [NVIDIA Jetson Orin Nano Super Developer Kit][jetson-link]. While capable, its 8GB unified memory limited me to ~8B parameter models, which produced subpar summaries.

I recently upgraded to an [ASUS Ascent GX10][gx10-link], a lower-cost alternative to the [NVIDIA DGX Spark][spark-link]. With 128GB of unified memory, I can now run much larger models. I am currently running a 30B parameter model (quantized) with excellent results. Theoretically, the hardware supports models up to 200B parameters.

## Dependencies

Locsum requires the following external libraries:

* **[ollama][ollama-link]:** Used for text summarization
* **[openai-whisper][whisper-link]:** Used for audio transcription

These libraries and their sub-dependencies will be installed automatically when you install Locsum.

## Installation

### Whisper Installation

Here is how I installed Whisper on my GX10. The exact steps may differ on your system.

- **Prerequisites:** Ensure `ffmpeg` is installed on your system
- **Get the CUDA version:** Run `nvidia-smi` to check your driver version (13.0 in my case)
- **Install [PyTorch][pytorch-link] and [Whisper][whisper-link]:** Create a [virtual environment][venv-link] and install the CUDA 13.0 build of PyTorch (only `torch` is required, not `torchvision`)

```sh
python3 -m venv ~/.local/venvs/whisper
source ~/.local/venvs/whisper/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/cu130
pip3 install openai-whisper
```
- **Verify installation:** Test CUDA is available in PyTorch and load the smallest Whisper model
```python
python3
>>> import torch
>>> import whisper
>>> torch.cuda.is_available()
>>> model = whisper.load_model("tiny").to("cuda")
```

### Ollama Installation

Coming soon.

### Locsum Installation with `pipx`

It is recommended to install Locsum within a [virtual environment][venv-link] to avoid conflicts with system packages. Some Linux distributions enforce this. You can use `pipx` to handle the virtual environment automatically, or create one manually and use `pip`.

`pipx` installs Locsum in an isolated environment and makes it available globally.

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

**2. Install Locsum:**

```bash
pipx install locsum
```

### Locsum Installation with `pip`

If you prefer to manage the virtual environment manually, you can create and activate it by following this [tutorial][venv-link]. Then install Locsum:

```bash
pip install locsum
```

## Deployments

View all releases on:

- **[PyPI Releases][pypi-releases]**
- **[GitHub Releases][github-releases]**

## Usage

### Basic Usage

```bash
locsum [arguments] FILE [FILE ...]
```

### Command-Line Arguments

Coming soon.

## Configuration

(Not yet implemented.)

When you run Locsum for the first time, a `config.toml` file is automatically created. Its location depends on your operating system (typical paths are listed below):

- **Linux:** `~/.config/locsum`
- **macOS:** `~/Library/Preferences/locsum`
- **Windows:** `C:/Users/YourUsername/AppData/Roaming/locsum`

You can edit this file to customize various settings. Common customizations include....

## VPN Setup

Since the goal is to process files locally, we might as well download them as privately as possible. Here is how I installed and configured WireGuard VPN on my GX10.

First update your system with `sudo apt update && sudo apt upgrade`. If the kernel is updated during this step, a reboot is required before continuing.

- Install WireGuard: `sudo apt install wireguard`
- Download WireGuard configuration from my [Proton VPN][proton-link] account
- Copy the configuration file to `/etc/wireguard/protonvpn.conf` and `chown root:root` (with sudo)
- Test connection manually
  - Connect: `sudo wg-quick up protonvpn`
  - Check connection: `sudo wg`
  - Check IP address: `curl -4 ip.me`
  - Disconnect: `sudo wg-quick down protonvpn`
- Connect at boot: `sudo systemctl enable --now wg-quick@protonvpn.service`
- Reboot and check VPN connection / IP address

## Radio Deactivation

For a truly air-gapped system and to eliminate electromagnetic radiation, here is how to disable the antennas:

- **Disable Bluetooth:**

```sh
sudo systemctl disable --now bluetooth
sudo rfkill block bluetooth
```

- **Disable wifi:**

```sh
sudo systemctl disable --now wpa_supplicant
sudo rfkill block wifi
nmcli radio wifi off
```

- **Check:**

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

Thanks to the creators and contributors of the powerful [openai-whisper][whisper-link] and [ollama][ollama-link] libraries for making this project possible.

[github-releases]: https://github.com/monsieurlinux/locsum/releases
[gx10-link]: https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/
[jetson-link]: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
[license-badge]: https://img.shields.io/pypi/l/locsum.svg
[license-link]: https://github.com/monsieurlinux/locsum/blob/main/LICENSE
[ollama-link]: https://github.com/ollama/ollama-python
[proton-link]: https://protonvpn.com/
[pypi-releases]: https://pypi.org/project/locsum/#history
[pypi-badge]: https://img.shields.io/pypi/v/locsum.svg
[pypi-link]: https://pypi.org/project/locsum/
[pytorch-link]: https://pytorch.org/get-started/locally/
[spark-link]: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
[venv-link]: https://docs.python.org/3/tutorial/venv.html
[whisper-link]: https://github.com/openai/whisper

