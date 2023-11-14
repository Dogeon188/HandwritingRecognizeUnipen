import sys
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
import keras

from config import no_cap

height_width = 64
scale = 5


class Recognizer:
    _instance = None

    @staticmethod
    def get_instance():
        if Recognizer._instance is None:
            Recognizer._instance = Recognizer()
        return Recognizer._instance
    
    def __init__(self):
        if Recognizer._instance is not None:
            raise Exception("Only one instance of Recognizer is allowed.")
        target_model = "unipen_no_cap_model" if no_cap else "unipen_model"
        print(f"Loading model {target_model}...")
        self.model = keras.models.load_model(f"data/{target_model}.h5")
        print("Model loaded.")
    
    def predict(self, bitmap):
        # bitmap = np.expand_dims(bitmap, axis=0)
        return self.model.predict(bitmap, verbose=0)


class Canvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(height_width * scale, height_width * scale)
        pixmap.fill(Qt.black)
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.pen = (QtGui.QColor('white'), 12)

    def set_pen(self, color, width):
        self.pen = (QtGui.QColor(color), width)

    def clear_canvas(self):
        self.pixmap().fill(Qt.black)
        self.update()

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setColor(self.pen[0])
        p.setWidth(self.pen[1])
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
        self.recognize()

    def recognize(self):
        img = self.pixmap().toImage()
        bitmap = np.frombuffer(img.bits().asarray(img.sizeInBytes()), np.uint8)
        bitmap = bitmap.reshape((height_width * scale, height_width * scale, 4))[:, :, 0]
        bitmap = bitmap.reshape(64, 5, 64, 5).mean(axis=(1, 3))
        res = Recognizer.get_instance().predict(np.expand_dims(bitmap, axis=0)).argmax()
        self.window().centralWidget().layout().itemAt(1).widget().setText(chr(res + 32))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas()

        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.canvas)

        rec = QtWidgets.QLabel("Write anything to start!")
        rec.setAlignment(Qt.AlignCenter)
        rec.setFont(QtGui.QFont("Arial", 24))
        l.addWidget(rec)

        palette = QtWidgets.QHBoxLayout()
        
        btn_pen = QtWidgets.QPushButton("Pen")
        btn_pen.pressed.connect(lambda: self.canvas.set_pen("#FFFFFF", 12))
        palette.addWidget(btn_pen)
        btn_erase = QtWidgets.QPushButton("Erase")
        btn_erase.pressed.connect(lambda: self.canvas.set_pen("#000000", 48))
        palette.addWidget(btn_erase)
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.pressed.connect(lambda: self.canvas.clear_canvas())
        palette.addWidget(btn_clear)

        l.addLayout(palette)
        self.setCentralWidget(w)

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()