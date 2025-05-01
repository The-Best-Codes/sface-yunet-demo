## SFace + Yunet Example

This is an example of using SFace + Yunet models for face detection and recognition.

## Installation

### You will need:

- Python (preferably Python 3.11)
- A WebCam

### Download & Run

> [!NOTE]
> It is recommended to use [`uv`](https://github.com/astral-sh/uv) for installing packages and running the application, but you can probably use `pip` instead.

Clone the repository:

```bash
git clone https://github.com/The-Best-Codes/sface-yunet-demo.git
```

Install dependencies:

```bash
cd sface-yunet-demo
uv add numpy opencv-python tqdm
```

## Usage

Run the application:

```bash
uv run main.py
```

- Save an image of yourself as a profile. Click the save button:

![Save Button Image](.assets/topbar_save_button.png)

- Save the image in the `data/images` directory in the project directory.
- Restart the program and it will recognize your face (as the file name you saved).
