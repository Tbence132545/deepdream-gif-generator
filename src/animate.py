import sys, os, numpy as np, tensorflow as tf
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QProgressBar, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
from PIL import Image
from tensorflow.keras.applications import VGG19

from deepdream import create_dream_model, gradient_ascent
from zoom import transform_image
from utils import save_gif


FRAME_DIR = "frames"
OUTPUT_FILE = "output/dream_animation.gif"
NUM_FRAMES = 500
ZOOM_FACTOR = 1.03
STEP_SIZE = 1.0
LAYER_NAMES = ['block5_conv1', 'block5_conv2', 'block5_conv3']
FRAME_DELAY = 33
RESIZE_DIM = 400

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

model = VGG19(weights='imagenet', include_top=False)
dream_model = create_dream_model(model, LAYER_NAMES)

class DreamWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Dream Animator")
        self.setFixedSize(500, 600)

        self.image_var = None
        self.frame_index = 0
        self.image_path = None

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setLayout(self.layout)

        self.start_button = QPushButton("Start Animation")
        self.start_button.setEnabled(False)
        self.start_button.setFixedWidth(200)
        self.start_button.clicked.connect(self.start_animation)
        self.layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel("Drag an image here")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border:2px solid black; background-color:#ccccff")
        self.layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, NUM_FRAMES)
        self.progress_bar.setFixedWidth(400)
        self.layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setAcceptDrops(True)
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_frame)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.image_path = event.mimeData().urls()[0].toLocalFile()
        pil_img = Image.open(self.image_path).convert("RGB").resize((RESIZE_DIM, RESIZE_DIM))
        qimg = QImage(pil_img.tobytes("raw", "RGB"), pil_img.width, pil_img.height, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        self.start_button.setEnabled(True)

    def start_animation(self):
        if not self.image_path:
            return
        pil_img = Image.open(self.image_path).convert("RGB").resize((RESIZE_DIM, RESIZE_DIM))
        img_array = np.expand_dims(np.array(pil_img, dtype=np.float32), axis=0)
        self.image_var = tf.Variable(img_array)
        self.frame_index = 0
        self.progress_bar.setValue(0)
        self.timer.start(FRAME_DELAY)
        self.start_button.setEnabled(False)

    def animate_frame(self):
        if self.frame_index >= NUM_FRAMES:
            self.timer.stop()
            save_gif(FRAME_DIR, OUTPUT_FILE, NUM_FRAMES, FRAME_DELAY)
            return

        self.image_var = gradient_ascent(self.image_var, dream_model, STEP_SIZE)

        img_np = self.image_var[0].numpy()
        img_np = transform_image(img_np, ZOOM_FACTOR)
        self.image_var.assign(np.expand_dims(img_np, axis=0))

        img_disp = np.clip(self.image_var[0].numpy(), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_disp)
        pil_img.save(os.path.join(FRAME_DIR, f"frame_{self.frame_index:04d}.png"))
        qimg = QImage(pil_img.tobytes("raw", "RGB"), pil_img.width, pil_img.height, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

        self.frame_index += 1
        self.progress_bar.setValue(self.frame_index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DreamWindow()
    window.show()
    sys.exit(app.exec())
