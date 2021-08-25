import sys
import cv2
from PyQt5.QtCore import QLine, QRectF
import numpy as np
from gui import Ui_InteractiveDraw
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtGui, QtCore
from detection_test import Detection_start
import signal


class Thread(QtCore.QThread, Detection_start):
    changePixmap = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QtCore.QThread.__init__(self, *args, **kwargs)
        self.flag = False

    def run(self):
        cap1 = cv2.VideoCapture(0)
        fps = cap1.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)
        # signal.signal(signal.SIGINT, self.signal_handler)

        if self.max_length < self.min_length:
            print("\nInvalid Arguments: Max length of the object is less then Min length!\n")
            sys.exit(0)

        if self.max_width < self.min_width:
            print("\nInvalid Arguments: Max width of the object is less then Min width!\n")
            sys.exit(0)

        self.flag = True
        while cap1.isOpened() and self.flag:
            ret, frame = cap1.read()
            if not ret:
                break
            frame, canny = self.rotated_rect(255 - cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), frame)
            displayFrame = frame.copy()
            self.add_image(frame)
            assemble_line = self.getcurrent_info()
            ex_length = str("[{} - {}]".format(self.min_length, self.max_length))
            ex_width = str("[{} - {}]".format(self.min_width, self.max_width))
            Measurement = "{}".format(assemble_line.area)
            Length = "{}".format(assemble_line.length)
            Width = "{}".format(assemble_line.width)
            total_part = "{}".format(self.total_parts)
            total_defects = "{}".format(self.total_defect)
            defects = "{}".format("False")

            if assemble_line.show is True:
                cv2.rectangle(displayFrame, (assemble_line.rects[0], assemble_line.rects[1]), (
                    assemble_line.rects[0] + assemble_line.rects[2], assemble_line.rects[1] + assemble_line.rects[3]),
                              (0, 0, 255), 2)
                defects = "Defect : {}".format("True")
                cv2.putText(displayFrame, defects, (5, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            else:
                if assemble_line.rects:
                    cv2.rectangle(displayFrame, (assemble_line.rects[0], assemble_line.rects[1]), (
                        assemble_line.rects[0] + assemble_line.rects[2],
                        assemble_line.rects[1] + assemble_line.rects[3]),
                                  (0, 255, 0), 2)
                    defects = "Defect : {}".format("False")
                    cv2.putText(displayFrame, defects, (5, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            displayFrame = cv2.cvtColor(displayFrame, cv2.COLOR_BGR2RGB)
            cvt2qt = QtGui.QImage(displayFrame.data, displayFrame.shape[1], displayFrame.shape[0],
                                  QtGui.QImage.Format_RGB888)
            canny_qt = QtGui.QImage(canny.data, canny.shape[1], canny.shape[0], QtGui.QImage.Format_Grayscale8)
            txt = {"frame": cvt2qt, "canny": canny_qt, "total_part": total_part, "total_defects": total_defects,
                   "area": Measurement, "length": Length,
                   "width": Width, "defects": defects, "ex_length": ex_length, "ex_width": ex_width}
            self.changePixmap.emit(txt)
            if cv2.waitKey(delay) > 0:
                break

    def stop(self):
        self.flag = False


class ImageScroller(QWidget):

    def __init__(self):
        self.chosen_points = []
        QWidget.__init__(self)
        self.pos = None
        self.setMouseTracking(True)
        self.start_y = None
        self.start_x = None
        self.end_x = None
        self.end_y = None
        self.clicked = False
        self.distance_from_center = None
        self.color = [0, 0, 0]
        self.line_width = 1
        self.text_width = 1
        self.lines = []
        self.label_list = []
        self.start = False

    def mousePressEvent(self, event):
        if event.button() == 1:
            self.start_x = event.x()
            self.start_y = event.y()
            self.clicked = True

    def mouseMoveEvent(self, event):
        if self.clicked and self.start:
            self.pos = event.pos()
            self.end_x = event.x()
            self.end_y = event.y()
            self.distance_from_center = round(
                ((event.y() - self.start_x) ** 2 + (event.x() - self.start_y) ** 2) ** 0.5)
            self.update()

    def paintEvent(self, event):
        if self.pos and self.start:

            painter = QPainter(self)
            # currentBrush = QBrush(painter.brush())
            # painter.setBrush(currentBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(self.color[0], self.color[1], self.color[2]), self.line_width,
                                      QtCore.Qt.SolidLine))
            painter.drawLine(self.end_x, self.end_y, self.start_x, self.start_y)
            font = QFont()
            font.setPointSize(self.text_width)
            painter.setFont(font)
            text_x = (self.end_x - self.start_x) / 2 + self.start_x + 20
            text_y = (self.end_y - self.start_y) / 2 + self.start_y + 20
            painter.drawText(text_x, text_y, "Distance:: " + str(self.distance_from_center) + " mm")
            for i in range(len(self.lines)):
                x0, y0, x1, y1 = self.lines[i][0], self.lines[i][1], self.lines[i][2], self.lines[i][3]
                painter.drawLine(x0, y0, x1, y1)
                dx = (x1 - x0) / 2 + x0 + 20
                dy = (y1 - y0) / 2 + y0 + 20
                painter.drawText(dx, dy, self.label_list[i])
        else:
            self.lines.clear()
            self.label_list.clear()
            # self.start_x, self.start_y, self.end_x, self.end_x = 0, 0, 0, 0

    def mouseReleaseEvent(self, event):
        self.clicked = False
        line_label = "Distance:: " + str(
            round(((event.y() - self.start_x) ** 2 + (event.x() - self.start_y) ** 2) ** 0.5)) + " mm"
        line = [self.start_x, self.start_y, event.x(), event.y()]
        self.lines.append(line)
        self.label_list.append(line_label)


class interactive(QMainWindow, Ui_InteractiveDraw):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.th = Thread(self)

        self.th.changePixmap.connect(self.setImage)
        self.th.start()
        self.w = ImageScroller()
        self.verticalLayout_2.addWidget(self.w)
        self.pos = None
        self.setMouseTracking(True)
        self.lines = []
        self.label_list = []
        self.slider_red.setValue(255)
        self.slider_green.setValue(255)
        self.slider_blue.setValue(255)
        self.set_color()
        self.set_width()
        self.slider_red.valueChanged.connect(self.set_color)
        self.slider_green.valueChanged.connect(self.set_color)
        self.slider_blue.valueChanged.connect(self.set_color)
        self.slider_line_width.valueChanged.connect(self.set_width)
        self.slider_text_width.valueChanged.connect(self.set_width)
        self.btn_start.clicked.connect(self.btn_click)

    def btn_click(self):
        self.w.start = self.btn_start.isChecked()
        if self.btn_start.isChecked():
            self.btn_start.setStyleSheet("QPushButton"
                                         "{"
                                         "background-color : rgb(0,255,255);"
                                         "border :2px solid ;"
                                         "border-color: rgb(255,255,255);"
                                         "}"
                                         )
        else:
            self.btn_start.setStyleSheet("QPushButton"
                                         "{"
                                         "background-color : rgb(0,0,0,0);"
                                         "border :2px solid ;"
                                         "border-color: rgb(255,255,255);"
                                         "}"
                                         )

    def set_width(self):
        self.w.line_width = self.slider_line_width.value()
        self.w.text_width = self.slider_text_width.value()

    def set_color(self):
        self.w.color = np.array([self.slider_red.value(), self.slider_green.value(), self.slider_blue.value()])
        style_color = "background-color:rgb(" + str(self.w.color[0]) + "," + str(self.w.color[1]) + "," + str(
            self.w.color[2]) + ")"
        self.color_preview.setStyleSheet(style_color)

    def image_show(self, image, plane):
        image = image.scaled(plane.width(), plane.height())
        palette = QtGui.QPalette()
        palette.setBrush(self.backgroundRole(), QtGui.QBrush(QPixmap(image)))
        plane.setAutoFillBackground(True)
        plane.setPalette(palette)
        plane.show()

    @QtCore.pyqtSlot(object)
    def setImage(self, txt):
        image = txt["frame"]
        self.lbl_width.setText(txt["width"])
        self.lbl_length.setText(txt["length"])
        self.lbl_area.setText(txt["area"])
        self.lbl_defects.setText(txt["defects"])
        self.lbl_ex_length.setText(txt["ex_length"])
        self.lbl_ex_width.setText(txt["ex_width"])
        self.lbl_total_defects.setText(txt["total_defects"])
        self.lbl_total_parts.setText(txt["total_part"])
        self.th.kernel_count1 = self.spinBox_kernel1.value()
        self.th.kernel_count2 = self.spinBox_kernel2.value()
        self.th.threshold_count = self.spinBox_threshold.value()
        self.image_show(image, self.w)
        canny_pixmap = QPixmap(txt["canny"])
        canny_pixmap = canny_pixmap.scaled(self.label_7.width(), self.label_7.height(), QtCore.Qt.KeepAspectRatio)
        self.label_7.setPixmap(canny_pixmap)

    def closeEvent(self, event):
        self.th.stop()
        self.th.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    isMain = True
    app = QApplication(sys.argv)
    window = interactive()
    window.show()
    sys.exit(app.exec_())
