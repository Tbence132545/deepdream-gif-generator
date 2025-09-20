# DeepDream gif generator

A user-friendly desktop application to generate DeepDream animations from images. This tool uses TensorFlow and a pre-trained VGG19 model to create visuals with a simple drag-and-drop interface.




https://github.com/user-attachments/assets/6bea964e-4b58-47c9-902c-bc8a1efb8c70




---

## Features

-   **Drag-and-Drop Interface**: Load images by dragging them into the application window.
-   **DeepDream Processing**: Utilizes frame-by-frame gradient ascent to generate visuals.
-   **Dynamic Effects**: Optional zooming and slight rotation.
-   **GIF Export**: Automatically saves processed frames and compiles them into a final GIF.
-   **Customizable Parameters**: Adjust layers, zoom factor, step size, frame count, and frame delay to fine-tune your output.

---

## How it works  
This tool generates visuals using the idea of DeepDream, a computer vision technique developed by Google that finds and enhances patterns in images. You can learn more about the process from Google AI's original blog post: https://research.google/blog/inceptionism-going-deeper-into-neural-networks/


## Project Structure

The project is organized as follows:  
/src  
-> /deepdream.py  
-> /utils.py  
-> /zoom.py  
-> /animate.py  

## Getting Started

### Prerequisites

* Python 3.8+
* `pip` package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tbence132545/deepdream-gif-generator
    cd deepdream-gif-generator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  **Run the application** from the project's root directory:
    ```bash
    python src/animate.py
    ```

2.  **Drag and drop** an image file onto the application window.

3.  Click the **"Start Animation"** button to begin the generation process.

Frames will be saved to the `frames/` folder, and the final animation will appear in the `output/` folder when finished.

---

## Configuration

You can adjust the animation parameters directly in `src/animate.py`. Key variables include:

-   `NUM_FRAMES`: Total number of frames for the animation.
-   `ZOOM_FACTOR`: The zoom multiplier for each frame.
-   `STEP_SIZE`: The learning rate for the gradient ascent.
-   `LAYER_NAMES`: A list of VGG19 layers to target.
-   `FRAME_DELAY`: The delay between GIF frames in milliseconds.

---

## License

This project is licensed under the MIT License.
  


