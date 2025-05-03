## SFace + Yunet Example

This is an example of using SFace + Yunet models for face detection and recognition.

## Installation

### You will need:

- Python (preferably Python 3.11)
- A WebCam

### Download Everything

### Clone Repo

> [!NOTE]
> It is recommended to use [`uv`](https://github.com/astral-sh/uv) for installing packages and running the application, but you can probably use `pip` instead.

Clone the repository:

```bash
git clone https://github.com/The-Best-Codes/sface-yunet-demo.git
```

### Download Model Files

> [!IMPORTANT]
> You **must** run the Git LFS commands below after pulling the repository to pull the model ONNX files. They are stored in Git LFS and won't be pulled by default.

Check and make sure you have Git LFS installed:

```bash
git lfs --version
```

If you don't, install it. You can check the [Git LFS Website](https://git-lfs.com/) for instructions.

Initialize Git LFS:

```bash
git lfs install
```

Now, pull the Git LFS model files:

```bash
git lfs pull
```

## Usage

Install dependencies:

```bash
cd sface-yunet-demo
uv add -r requirements.txt
```

Run the application:

```bash
uv run main.py
```

- Save an image of yourself as a profile. Click the save button:

![Save Button Image](.assets/topbar_save_button.png)

- Save the image in the `data/images` directory in the project directory.
- Restart the program and it will recognize your face (as the file name you saved).
