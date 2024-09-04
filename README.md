
# Image Upload and Detection with FastAPI and YOLO
## this project is not yet completed just initial work there is bit more work is ther which you can see in the nutrition.ipynb and also visuvalisation part also left.
This project allows users to upload an image via a web interface, sends the image to a FastAPI server for object detection using YOLO, and displays the detection results along with the processed image.

## Features

- File upload through a web interface.
- Object detection using YOLO model on the server side.
- Display of detection results and processed image on the web interface.

## Prerequisites

- Python 3.7 or higher
- FastAPI
- Uvicorn
- CORS Middleware
- OpenCV
- NumPy
- PIL (Python Imaging Library)
- ultralytics (YOLO implementation)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/PonnaSrikar/final-nutrivision.git
    cd final-nutrivision
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required Python packages:**
    ```sh
    pip install fastapi uvicorn numpy opencv-python pillow ultralytics
    ```

4. **Save the YOLO model file (`best.pt`) in the project directory:**
    - Ensure that the YOLO model file (`best.pt`) is in the same directory as the FastAPI server script.

## Running the Server

1. **Start the FastAPI server:**
    ```sh
    python server.py
    ```

    The server will be running at `http://127.0.0.1:8000`.

## Using the Web Interface

1. **Open the `index.html` file in your web browser:**
    - Double-click the `index.html` file or open it using your preferred method.

2. **Upload an image:**
    - Click on the "Choose File" button to select an image file.
    - Click on the "Upload" button to send the image to the server.

3. **View the results:**
    - The detection results will be displayed in a table below the upload form.
    - The processed image with detected objects will be displayed below the table.

## Project Structure

