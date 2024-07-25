import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
import tensorflow as tf
import sympy as sp

class MathApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math Equation Solver")
        self.setGeometry(100, 100, 600, 400)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()

        layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setPixmap(QPixmap.fromImage(self.image))
        layout.addWidget(self.label)

        self.button = QPushButton('Calculate', self)
        self.button.clicked.connect(self.calculate)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.label.setPixmap(QPixmap.fromImage(self.image))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False


    def calculate(self):

        buffer = self.image.bits().asstring(self.image.byteCount())
        image = np.frombuffer(buffer, np.uint8).reshape(self.image.height(), self.image.width(), 4)
        equation = recognize_handwriting(image)
        result = solve_equation(equation)
        self.label.setText(f"{equation} = {result}")
        self.update()

def recognize_handwriting(image):
    # Preprocess the image for the model
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 32))  # Example size, adjust as necessary
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)

    # Load the pre-trained model
    model = tf.keras.models.load_model('./model.h5')

    # Predict the equation
    predictions = model.predict(input_data)
    equation = decode_predictions(predictions)
    return equation

def decode_predictions(predictions):
    # Implement decoding of model predictions to equation string
    pass

def solve_equation(equation):
    try:
        result = sp.sympify(equation)
        solved = sp.solve(result)
        return solved
    except sp.SympifyError:
        return "Error solving equation"
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MathApp()
    window.show()
    sys.exit(app.exec_())
