import sys
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
import numpy as np
import tensorflow as tf
import os

height_width = 64
scale = 5

data_path = "data/curated"

def path2label(p):
    label_id = tf.strings.split(p, os.path.sep)[2]
    label_id = tf.strings.to_number(label_id, out_type=tf.int32)
    return label_id - 32

print("Loading dataset informations...")
dataset = tf.data.Dataset.list_files(os.path.join(data_path, "*/*.png")).map(path2label)
arr = np.fromiter(tf.data.Dataset.as_numpy_iterator(dataset), np.int32)
counts = np.bincount(arr)
arrsize = arr.size

rng = np.random.default_rng()

def next_target():
    global counts, rng
    prob = 1 / counts
    prob[0] = 0.000001
    prob = prob / prob.sum()
    target = rng.choice(prob.size, p=prob) + 32
    return target

target = next_target()

class Canvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(height_width * scale, height_width * scale)
        pixmap.fill(Qt.black)
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.setPenPen()
    
    def getResultLabel(self):
        return self.window().centralWidget().layout().itemAt(1).widget()
    
    def getSaveLabel(self):
        return self.window().centralWidget().layout().itemAt(2).widget()
    
    def setPenPen(self):
        self.pen = (QtGui.QColor('white'), 5)
        self.setCursor(Qt.CrossCursor)
    
    def setPenEraser(self):
        self.pen = (QtGui.QColor('black'), 20)
        self.setCursor(Qt.UpArrowCursor)

    def clearCanvas(self):
        self.pixmap().fill(Qt.black)
        self.update()
    
    def saveCanvas(self):
        global arrsize, target
        self.pixmap().scaled(64, 64).save(f"data/curated/{target}/{arrsize+100000}.png")
        self.getSaveLabel().setText(f"Saved to 'data/curated/{target}/{arrsize+100000}.png'")
        self.clearCanvas()
        target = next_target()
        self.getResultLabel().setText(f"Please draw: '{chr(target)}' ({target})")
        counts[target - 32] += 1
        arrsize += 1

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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas()
        self.canvas.setAlignment(Qt.AlignCenter)

        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.canvas)

        rec = QtWidgets.QLabel(f"Please draw: '{chr(target)}' ({target})")
        rec.setAlignment(Qt.AlignCenter)
        monofont = QtGui.QFont("Monospace", 20)
        monofont.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        rec.setFont(monofont)
        l.addWidget(rec)

        sav = QtWidgets.QLabel("Write something and press 'Save'")
        sav.setAlignment(Qt.AlignCenter)
        monofont_s = QtGui.QFont("Monospace", 12)
        monofont_s.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        sav.setFont(monofont_s)
        l.addWidget(sav)

        btns1 = QtWidgets.QHBoxLayout()

        btn_pen = QtWidgets.QPushButton("Pen (P)")
        btn_pen.pressed.connect(self.canvas.setPenPen)
        btns1.addWidget(btn_pen)
        btn_erase = QtWidgets.QPushButton("Erase (R)")
        btn_erase.pressed.connect(self.canvas.setPenEraser)
        btns1.addWidget(btn_erase)

        l.addLayout(btns1)

        btns2 = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("Clear (C)")
        btn_clear.pressed.connect(self.canvas.clearCanvas)
        btns2.addWidget(btn_clear)
        btn_save = QtWidgets.QPushButton("Save (Spc/Ent)")
        btn_save.pressed.connect(self.canvas.saveCanvas)
        btns2.addWidget(btn_save)

        l.addLayout(btns2)

        self.setCentralWidget(w)
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_P:
            self.canvas.setPenPen()
        elif e.key() == Qt.Key_R:
            self.canvas.setPenEraser()
        elif e.key() == Qt.Key_C:
            self.canvas.clearCanvas()
        elif e.key() == Qt.Key_Return or e.key() == Qt.Key_Enter or e.key() == Qt.Key_Space:
            self.canvas.saveCanvas()
        else:
            super().keyPressEvent(e)

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
