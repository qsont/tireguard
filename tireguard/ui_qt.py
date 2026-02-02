# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import time
from dataclasses import asdict

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QComboBox, QTextEdit, QListWidget, QSplitter,
    QFormLayout, QMessageBox, QDockWidget, QSpinBox, QStackedWidget, QFrame, QScrollArea, QSizePolicy
)
try:
    from PySide6.QtWidgets import QScroller
except Exception:
    QScroller = None

from .config import APP_NAME, RES_PRESETS
from .camera import open_camera
from .preprocess import preprocess_bgr, crop_roi
from .quality import run_quality_checks
from .measure import groove_visibility_score, pass_fail_from_score
from .auto_roi import suggest_roi
from .calibration import load_calibration, save_calibration, compute_scale_from_two_points
from .live_metrics import compute_live_metrics

from .storage import (
    init_db, save_capture, save_processed, insert_result, export_csv,
    list_results, get_result_by_ts, find_processed_images
)

# ---------- Helpers ----------
def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

def clamp_roi(roi, frame_shape):
    fh, fw = frame_shape[:2]
    x = max(0, min(fw-1, int(roi["x"])))
    y = max(0, min(fh-1, int(roi["y"])))
    w = max(1, min(fw-x, int(roi["w"])))
    h = max(1, min(fh-y, int(roi["h"])))
    return {"x": x, "y": y, "w": w, "h": h}

# ---------- Toast ----------
class Toast(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("""
            QWidget {
                background: rgba(10, 16, 22, 0.92);
                border: 1px solid rgba(120, 220, 255, 0.35);
                border-radius: 14px;
            }
            QLabel { color: #e8f7ff; font-weight: 600; }
        """)
        self.label = QLabel("", self)
        self.label.setWordWrap(True)
        lay = QVBoxLayout()
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(self.label)
        self.setLayout(lay)

        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.hide()

    def show_toast(self, text, ms=1800):
        self.label.setText(text)
        self.adjustSize()

        parent = self.parentWidget()
        if not parent:
            return

        w = min(420, max(220, self.width()))
        h = self.height()
        margin = 18
        x = parent.width() - w - margin
        y0 = margin - 10
        y1 = margin

        self.setGeometry(x, y0, w, h)
        self.show()

        self.anim.stop()
        self.anim.setDuration(220)
        self.anim.setStartValue(QRect(x, y0, w, h))
        self.anim.setEndValue(QRect(x, y1, w, h))
        self.anim.start()

        QTimer.singleShot(ms, self._hide)

    def _hide(self):
        self.hide()

# ---------- Status Chip ----------
class StatusChip(QLabel):
    def __init__(self, text="READY"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(28)
        self.setStyleSheet("border-radius: 14px; padding: 4px 10px; font-weight: 800; letter-spacing: 0.5px;")
        self.set_state("ready")

    def set_state(self, state: str, text: str | None = None):
        # states: ok, warn, fail, ready
        if text is not None:
            self.setText(text)

        if state == "ok":
            self.setStyleSheet(self.styleSheet() + "background: rgba(0,255,170,0.14); border: 1px solid rgba(0,255,170,0.45); color: #7fffd4;")
        elif state == "warn":
            self.setStyleSheet(self.styleSheet() + "background: rgba(255,210,70,0.14); border: 1px solid rgba(255,210,70,0.45); color: #ffe08a;")
        elif state == "fail":
            self.setStyleSheet(self.styleSheet() + "background: rgba(255,70,120,0.14); border: 1px solid rgba(255,70,120,0.45); color: #ff8ab3;")
        else:
            self.setStyleSheet(self.styleSheet() + "background: rgba(120,220,255,0.10); border: 1px solid rgba(120,220,255,0.35); color: #9bdcff;")

# ---------- Video ----------
class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(820, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0b0f14; border-radius:18px; border: 1px solid rgba(120,220,255,0.18);")
        self.frame_bgr = None

        self.roi = None
        self.roi_mode = False
        self.dragging = False
        self.drag_start = None
        self.drag_end = None

        self.calib_mode = False
        self.calib_points = []

    def set_modes(self, roi_mode=False, calib_mode=False):
        self.roi_mode = roi_mode
        self.calib_mode = calib_mode
        if not calib_mode:
            self.calib_points = []
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.update()

    def set_frame(self, frame_bgr, roi=None):
        self.frame_bgr = frame_bgr
        if roi is not None:
            self.roi = roi
        self.update()

    def mousePressEvent(self, e):
        if self.frame_bgr is None:
            return
        if self.calib_mode:
            self.calib_points.append((e.position().x(), e.position().y()))
            if len(self.calib_points) > 2:
                self.calib_points = self.calib_points[-2:]
            self.update()
            w = self.window()
            if hasattr(w, 'on_calib_clicks_changed'):
                w.on_calib_clicks_changed()
            return
        if not self.roi_mode:
            return
        self.dragging = True
        self.drag_start = (e.position().x(), e.position().y())
        self.drag_end = self.drag_start
        self.update()

    def mouseMoveEvent(self, e):
        if not self.dragging:
            return
        self.drag_end = (e.position().x(), e.position().y())
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.dragging:
            return
        self.dragging = False
        self.drag_end = (e.position().x(), e.position().y())
        self.update()
        w = self.window();
        if hasattr(w, 'on_roi_drag_finished'):
            w.on_roi_drag_finished()

    def _widget_to_frame(self, x, y):
        if self.frame_bgr is None:
            return 0, 0
        fh, fw = self.frame_bgr.shape[:2]
        ww = max(1, self.width())
        wh = max(1, self.height())
        scale = min(ww / fw, wh / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        ox = (ww - disp_w) // 2
        oy = (wh - disp_h) // 2

        x = int((x - ox) / scale)
        y = int((y - oy) / scale)
        x = max(0, min(fw - 1, x))
        y = max(0, min(fh - 1, y))
        return x, y

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.frame_bgr is None:
            return

        qimg = bgr_to_qimage(self.frame_bgr)
        pix = QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # map frame->widget overlay via aspect-fit math
        fh, fw = self.frame_bgr.shape[:2]
        ww = max(1, self.width())
        wh = max(1, self.height())
        scale = min(ww / fw, wh / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        ox = (ww - disp_w) // 2
        oy = (wh - disp_h) // 2

        def f2w(px, py):
            return ox + int(px * scale), oy + int(py * scale)

        # saved ROI
        if self.roi:
            x, y, w, h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
            x0, y0 = f2w(x, y)
            x1, y1 = f2w(x + w, y + h)
            pen = QPen(QColor("#00ffb2"), 3)
            painter.setPen(pen)
            painter.drawRoundedRect(x0, y0, x1-x0, y1-y0, 14, 14)

        # drag ROI preview
        if self.roi_mode and self.drag_start and self.drag_end:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_end
            pen = QPen(QColor("#7bdcff"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1-x0)), int(abs(y1-y0)))

        # calib overlay
        if self.calib_mode and self.calib_points:
            pen = QPen(QColor("#7bdcff"), 3)
            painter.setPen(pen)
            for (x, y) in self.calib_points:
                painter.drawEllipse(int(x)-7, int(y)-7, 14, 14)
            if len(self.calib_points) == 2:
                p0, p1 = self.calib_points
                painter.drawLine(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))

        painter.end()

# ---------- Main (Wizard) ----------
class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        init_db(cfg)

        self.setWindowTitle(APP_NAME)
        self.resize(1360, 800)
        self._apply_theme()

        self.cap = None
        self.cam_index = None

        self.roi = None
        self.calib = load_calibration(cfg.calibration_path)

        # Live ROI metrics state
        self.prev_roi_gray = None
        self.live_last = None
        self.auto_trigger_enabled = False
        self.stable_ok_frames = 0
        self._metrics_last_t = 0.0

        self.toast = Toast(self)

        self._build_ui()
        self._open_camera()
        self._refresh_history()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(15)

    def _apply_theme(self):
        # Cyberpunk-ish accent teal/purple
        self.setStyleSheet("""
            QMainWindow { background: #070a10; }
            QLabel { color: #e8f7ff; }
            QPushButton {
                background: rgba(20, 28, 40, 0.9);
                color: #e8f7ff;
                border: 1px solid rgba(120, 220, 255, 0.22);
                padding: 12px;
                border-radius: 16px;
                font-weight: 700;
            }
            QPushButton:hover { background: rgba(30, 44, 66, 0.95); border: 1px solid rgba(120, 220, 255, 0.35); }
            QPushButton:pressed { background: rgba(12, 18, 28, 1.0); }
            QLineEdit, QComboBox, QTextEdit {
                background: rgba(12, 18, 28, 0.95);
                color: #e8f7ff;
                border: 1px solid rgba(120, 220, 255, 0.22);
                border-radius: 14px;
                padding: 8px;
            }
            QGroupBox {
                color: #e8f7ff;
                border: 1px solid rgba(120, 220, 255, 0.18);
                border-radius: 18px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; font-weight: 800; }
            QListWidget {
                background: rgba(12, 18, 28, 0.95);
                border: 1px solid rgba(120, 220, 255, 0.18);
                border-radius: 14px;
                color: #e8f7ff;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 6px 4px 6px 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(120, 220, 255, 0.25);
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(120, 220, 255, 0.40);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Top bar: title + chip
        top = QHBoxLayout()
        title = QLabel("TireGuard")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        subtitle = QLabel("Tread visibility & capture workflow")
        subtitle.setStyleSheet("color: rgba(232,247,255,0.65);")

        title_box = QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.chip = StatusChip("READY")
        self.chip.set_state("ready", "READY")

        top.addLayout(title_box)
        top.addStretch(1)
        top.addWidget(self.chip)

        # Main: split video + wizard panel
        self.video = VideoWidget()
        self.video.setParent(self)

        self.steps = QStackedWidget()

        # Step 1: Camera
        self.steps.addWidget(self._step_camera())
        # Step 2: ROI
        self.steps.addWidget(self._step_roi())
        # Step 3: Calibrate
        self.steps.addWidget(self._step_calibrate())
        # Step 4: Scan
        self.steps.addWidget(self._step_scan())

        self.btn_back = QPushButton("← Back")
        self.btn_next = QPushButton("Next →")
        self.btn_back.clicked.connect(self.prev_step)
        self.btn_next.clicked.connect(self.next_step)

        nav = QHBoxLayout()
        nav.addWidget(self.btn_back)
        nav.addWidget(self.btn_next)

        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.addWidget(self.steps)
        right_layout.addLayout(nav)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Make right panel scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(right_content)

        # Smooth/kinetic scrolling (feels closer to mobile/browser)
        if QScroller is not None:
            QScroller.grabGesture(scroll.viewport(), QScroller.LeftMouseButtonGesture)

        right_wrap = QWidget()
        right_outer = QVBoxLayout(right_wrap)
        right_outer.addWidget(scroll)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_wrap.setMinimumWidth(420)

        # Make sizing behavior nicer
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_wrap.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(right_wrap)

        # Prevent collapsing/shrinking weirdness
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.setStretchFactor(0, 3)   # video grows most
        splitter.setStretchFactor(1, 1)   # right grows some
        splitter.setSizes([900, 460])

        self.video.setMinimumWidth(640)
        right_wrap.setMinimumWidth(420)

        # Video should keep space; right panel can shrink/scroll
        splitter.setStretchFactor(0, 3)   # left (video)
        splitter.setStretchFactor(1, 1)   # right (panel)

        # Don’t let the video collapse too small
        splitter.setCollapsible(0, False)

        # It’s okay if the panel collapses smaller (since it scrolls)
        splitter.setCollapsible(1, True)

        # Optional: make handle easier to grab
        splitter.setHandleWidth(8)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(splitter)
        central.setLayout(layout)

        # Advanced dock
        self._build_advanced_dock()

        # menu
        act_adv = QAction("Toggle Advanced", self)
        act_adv.triggered.connect(self.toggle_advanced)
        self.menuBar().addAction(act_adv)

        self._update_nav_buttons()

    def _card(self, title_text: str, body_widget: QWidget) -> QWidget:
        box = QGroupBox(title_text)
        v = QVBoxLayout()
        v.addWidget(body_widget)
        box.setLayout(v)
        return box

    def _step_camera(self):
        w = QWidget()
        v = QVBoxLayout()

        hint = QLabel("Step 1: Choose camera & resolution")
        hint.setStyleSheet("color: rgba(232,247,255,0.78); font-weight:700;")
        v.addWidget(hint)

        form = QWidget()
        f = QFormLayout()

        self.cam_idx = QComboBox()
        self.cam_idx.addItems([str(i) for i in range(0, 6)])
        self.cam_idx.setCurrentText("0")

        self.res_combo = QComboBox()
        self.res_combo.addItems([p[0] for p in RES_PRESETS])
        self.res_combo.setCurrentText(f"{self.cfg.width}x{self.cfg.height}")

        self.btn_open_cam = QPushButton("Open camera")
        self.btn_open_cam.clicked.connect(self.reopen_camera)

        f.addRow("Camera index", self.cam_idx)
        f.addRow("Resolution", self.res_combo)
        f.addRow(self.btn_open_cam)
        form.setLayout(f)

        v.addWidget(self._card("Camera", form))
        v.addStretch(1)
        w.setLayout(v)
        return w

    def _step_roi(self):
        w = QWidget()
        v = QVBoxLayout()
        hint = QLabel("Step 2: Set ROI (tread area)")
        hint.setStyleSheet("color: rgba(232,247,255,0.78); font-weight:700;")
        v.addWidget(hint)

        btns = QWidget()
        h = QHBoxLayout()
        self.btn_roi_mode = QPushButton("Drag ROI")
        self.btn_roi_mode.clicked.connect(self.toggle_roi_mode)
        self.btn_auto_roi = QPushButton("Auto ROI")
        self.btn_auto_roi.clicked.connect(self.auto_roi)
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        h.addWidget(self.btn_roi_mode)
        h.addWidget(self.btn_auto_roi)
        h.addWidget(self.btn_clear_roi)
        btns.setLayout(h)

        v.addWidget(self._card("ROI Tools", btns))

        tip = QLabel("Tip: drag a rectangle around the tire tread. Keep it tight.")
        tip.setStyleSheet("color: rgba(232,247,255,0.65);")
        v.addWidget(tip)
        v.addStretch(1)
        w.setLayout(v)
        return w

    def _step_calibrate(self):
        w = QWidget()
        v = QVBoxLayout()
        hint = QLabel("Step 3: (Optional) Calibration")
        hint.setStyleSheet("color: rgba(232,247,255,0.78); font-weight:700;")
        v.addWidget(hint)

        box = QWidget()
        b = QVBoxLayout()

        self.cal_label = QLabel(self._calib_text())
        self.btn_cal = QPushButton("Start 2-point calibration")
        self.btn_cal.clicked.connect(self.toggle_calibration)

        self.btn_clear_cal = QPushButton("Clear calibration")
        self.btn_clear_cal.clicked.connect(self.clear_calibration)

        b.addWidget(self.cal_label)
        b.addWidget(self.btn_cal)
        b.addWidget(self.btn_clear_cal)
        box.setLayout(b)

        v.addWidget(self._card("Scale (mm/px)", box))
        note = QLabel("Use a ruler or known-width object in the same plane as the tread.")
        note.setStyleSheet("color: rgba(232,247,255,0.65);")
        v.addWidget(note)
        v.addStretch(1)
        w.setLayout(v)
        return w

    def _step_scan(self):
        w = QWidget()
        v = QVBoxLayout()

        hint = QLabel("Step 4: Scan & Save")
        hint.setStyleSheet("color: rgba(232,247,255,0.78); font-weight:700;")
        v.addWidget(hint)

        # session fields
        sess = QWidget()
        f = QFormLayout()
        self.in_vehicle = QLineEdit()
        self.in_operator = QLineEdit()
        self.in_tirepos = QComboBox()
        self.in_tirepos.addItems(["FL","FR","RL","RR","SPARE"])
        self.in_notes = QLineEdit()

        f.addRow("Vehicle ID", self.in_vehicle)
        f.addRow("Tire", self.in_tirepos)
        f.addRow("Operator", self.in_operator)
        f.addRow("Notes", self.in_notes)
        sess.setLayout(f)
        v.addWidget(self._card("Session", sess))

        # Aim Assist (live meters)
        self.aim_text = QLabel("Aim Assist: set ROI to see live metrics")
        self.aim_text.setStyleSheet("color: rgba(232,247,255,0.75); font-weight:700;")
        self.aim_box = QTextEdit()
        self.aim_box.setReadOnly(True)
        self.aim_box.setMinimumHeight(120)
        v.addWidget(self._card("Aim Assist", self.aim_box))

        # Scan options
        opt = QWidget(); fo = QFormLayout()
        self.sp_burst = QSpinBox(); self.sp_burst.setRange(1, 25); self.sp_burst.setValue(8)
        self.sp_stable_need = QSpinBox(); self.sp_stable_need.setRange(1, 30); self.sp_stable_need.setValue(8)
        self.cb_auto_trigger = QComboBox(); self.cb_auto_trigger.addItems(["Off", "On"])
        fo.addRow("Burst frames", self.sp_burst)
        fo.addRow("Auto-trigger", self.cb_auto_trigger)
        fo.addRow("Stable frames needed", self.sp_stable_need)
        opt.setLayout(fo)
        v.addWidget(self._card("Scan Options", opt))

        self.btn_capture = QPushButton("Capture + Analyze")
        self.btn_capture.clicked.connect(self.capture_analyze)
        v.addWidget(self.btn_capture)

        self.cb_auto_trigger.currentTextChanged.connect(self._on_auto_trigger_changed)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.export_csv_action)
        v.addWidget(self.btn_export)

        # result
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(180)
        v.addWidget(self._card("Result", self.result))

        v.addStretch(1)
        w.setLayout(v)
        return w

    def _build_advanced_dock(self):
        self.adv_dock = QDockWidget("Advanced", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.adv_dock)
        self.adv_dock.setVisible(False)

        adv = QWidget()
        v = QVBoxLayout()

        # thresholds
        thr = QGroupBox("Thresholds")
        f = QFormLayout()
        self.sp_min_bright = QSpinBox(); self.sp_min_bright.setRange(0, 255); self.sp_min_bright.setValue(int(self.cfg.min_brightness))
        self.sp_max_bright = QSpinBox(); self.sp_max_bright.setRange(0, 255); self.sp_max_bright.setValue(int(self.cfg.max_brightness))
        self.sp_min_sharp = QSpinBox(); self.sp_min_sharp.setRange(0, 5000); self.sp_min_sharp.setValue(int(self.cfg.min_sharpness))
        btn_apply = QPushButton("Apply thresholds")
        btn_apply.clicked.connect(self.apply_thresholds)
        f.addRow("Min brightness", self.sp_min_bright)
        f.addRow("Max brightness", self.sp_max_bright)
        f.addRow("Min sharpness", self.sp_min_sharp)
        f.addRow(btn_apply)
        thr.setLayout(f)

        # history
        hist = QGroupBox("History")
        hv = QVBoxLayout()
        self.history = QListWidget()
        self.history.itemSelectionChanged.connect(self.on_history_select)
        hv.addWidget(self.history)
        hist.setLayout(hv)

        v.addWidget(thr)
        v.addWidget(hist)
        v.addStretch(1)
        adv.setLayout(v)
        self.adv_dock.setWidget(adv)

    # ---------- Wizard nav ----------
    def _update_nav_buttons(self):
        idx = self.steps.currentIndex()
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < self.steps.count() - 1)

        # smart labeling
        labels = ["Camera", "ROI", "Calibrate", "Scan"]
        self.btn_next.setText(("Next → " + labels[idx+1]) if idx < 3 else "Done")

    def next_step(self):
        idx = self.steps.currentIndex()
        # soft validation per step
        if idx == 0 and self.cap is None:
            self.toast.show_toast("Open a camera first.")
            return
        if idx == 1 and not self.roi:
            self.toast.show_toast("Set ROI (drag or auto) before continuing.")
            return
        if idx < self.steps.count() - 1:
            self.steps.setCurrentIndex(idx + 1)
            self._update_nav_buttons()

    def prev_step(self):
        idx = self.steps.currentIndex()
        if idx > 0:
            self.steps.setCurrentIndex(idx - 1)
            self._update_nav_buttons()

    # ---------- Camera ----------
    def _open_camera(self, force_index=None):
        pref = force_index if force_index is not None else self.cfg.cam_index
        self.cap, self.cam_index = open_camera(pref, self.cfg.width, self.cfg.height, self.cfg.fps)
        self.toast.show_toast(f"Camera opened (index={self.cam_index})")
        self.chip.set_state("ready", "READY")

    def reopen_camera(self):
        idx = int(self.cam_idx.currentText())
        preset = self.res_combo.currentText()
        for name, w, h in RES_PRESETS:
            if name == preset:
                self.cfg.width = w
                self.cfg.height = h
        if self.cap:
            self.cap.release()
        self._open_camera(force_index=idx)

    def _tick(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        self.video.set_frame(frame, roi=self.roi)

        # Live ROI metrics (Aim Assist + Auto-trigger)
        # useful for accurate detection and auto-trigger
        if self.roi is not None:
            t = time.monotonic()
            if (t - self._metrics_last_t) >= 0.10:
                self._metrics_last_t = t
                try:
                    roi_bgr = crop_roi(frame, self.roi)
                    live, roi_gray = compute_live_metrics(
                        roi_bgr,
                        self.prev_roi_gray,
                        min_brightness=self.cfg.min_brightness,
                        max_brightness=self.cfg.max_brightness,
                        min_sharpness=self.cfg.min_sharpness,
                    )
                    self.prev_roi_gray = roi_gray
                    self.live_last = live

                    # Update Aim Assist box if it exists
                    if hasattr(self, 'aim_box'):
                        msg = []
                        msg.append(f"Brightness: {live.brightness:.1f}  (target {self.cfg.min_brightness:.0f}-{self.cfg.max_brightness:.0f})")
                        msg.append(f"Sharpness:  {live.sharpness:.1f}  (min {self.cfg.min_sharpness:.0f})")
                        msg.append(f"Glare:      {live.glare_ratio*100:.1f}% (lower is better)")
                        msg.append(f"Stability:  {live.stability:.2f} (lower is better)")
                        msg.append("")
                        msg.append("Hint: " + ("READY TO CAPTURE" if live.ok_hint else "Adjust angle/light/hold steady"))
                        self.aim_box.setPlainText("\n".join(msg))

                    # Auto-trigger: only on Scan step
                    if self.auto_trigger_enabled and self.steps.currentIndex() == 3:
                        if live.ok_hint:
                            self.stable_ok_frames += 1
                        else:
                            self.stable_ok_frames = 0
                        need = self.sp_stable_need.value() if hasattr(self, 'sp_stable_need') else 8
                        if self.stable_ok_frames >= need:
                            self.stable_ok_frames = 0
                            self.toast.show_toast("Auto-trigger capture…")
                            self.capture_analyze()
                except Exception:
                    # avoid crashing the UI loop
                    pass

    # ---------- ROI ----------
    def toggle_roi_mode(self):
        self.video.set_modes(roi_mode=True, calib_mode=False)
        self.toast.show_toast("ROI mode: drag a rectangle on the video")

    def clear_roi(self):
        self.roi = None
        self.video.roi = None
        self.toast.show_toast("ROI cleared")

    def auto_roi(self):
        if self.video.frame_bgr is None:
            return
        self.roi = clamp_roi(suggest_roi(self.video.frame_bgr), self.video.frame_bgr.shape)
        self.video.set_frame(self.video.frame_bgr, roi=self.roi)
        self.toast.show_toast("Auto ROI set")

    def on_roi_drag_finished(self):
        if self.video.frame_bgr is None or not self.video.drag_start or not self.video.drag_end:
            return
        x0, y0 = self.video.drag_start
        x1, y1 = self.video.drag_end
        fx0, fy0 = self.video._widget_to_frame(x0, y0)
        fx1, fy1 = self.video._widget_to_frame(x1, y1)
        x = min(fx0, fx1); y = min(fy0, fy1)
        w = abs(fx1 - fx0); h = abs(fy1 - fy0)
        if w < 40 or h < 40:
            self.toast.show_toast("ROI too small. Try again.")
            return
        self.roi = clamp_roi({"x": x, "y": y, "w": w, "h": h}, self.video.frame_bgr.shape)
        self.video.set_modes(False, False)
        self.toast.show_toast(f"ROI set ({w}×{h})")

    # ---------- Calibration ----------
    def toggle_calibration(self):
        self.video.set_modes(roi_mode=False, calib_mode=True)
        self.toast.show_toast("Calibration: click 2 points with known distance")

    def on_calib_clicks_changed(self):
        if len(self.video.calib_points) == 2:
            from PySide6.QtWidgets import QInputDialog
            mm, ok = QInputDialog.getDouble(self, "Calibration", "Enter known distance (mm):", 10.0, 0.001, 10000.0, 3)
            if not ok:
                self.video.calib_points = []
                return
            p0 = self.video._widget_to_frame(*self.video.calib_points[0])
            p1 = self.video._widget_to_frame(*self.video.calib_points[1])
            px_per_mm, mm_per_px = compute_scale_from_two_points(p0, p1, float(mm))
            self.calib = {"px_per_mm": px_per_mm, "mm_per_px": mm_per_px, "method": "two_point"}
            save_calibration(self.cfg.calibration_path, self.calib)
            self.cal_label.setText(self._calib_text())
            self.video.set_modes(False, False)
            self.toast.show_toast("Calibration saved")

    def clear_calibration(self):
        self.calib = {"px_per_mm": None, "mm_per_px": None, "method": None}
        save_calibration(self.cfg.calibration_path, self.calib)
        self.cal_label.setText(self._calib_text())
        self.toast.show_toast("Calibration cleared")

    def _calib_text(self):
        if self.calib.get("mm_per_px"):
            return f"Scale: {self.calib['mm_per_px']:.6f} mm/px ({self.calib['px_per_mm']:.2f} px/mm)"
        return "Scale: not calibrated"

    # ---------- Thresholds ----------
    def apply_thresholds(self):
        self.cfg.min_brightness = float(self.sp_min_bright.value())
        self.cfg.max_brightness = float(self.sp_max_bright.value())
        self.cfg.min_sharpness = float(self.sp_min_sharp.value())
        self.toast.show_toast("Thresholds applied")

    # ---------- Capture ----------
    def capture_analyze(self):
        if self.video.frame_bgr is None:
            self.toast.show_toast("No camera frame.")
            return
        if not self.roi:
            self.toast.show_toast("Set ROI first.")
            return

        # Burst capture for reliability
        burst_n = self.sp_burst.value() if hasattr(self, 'sp_burst') else 1
        frames = []
        if burst_n <= 1:
            frame = self.video.frame_bgr.copy()
            frames = [frame]
        else:
            # grab burst_n frames from camera (fallback to last frame if needed)
            for _ in range(burst_n):
                ok, fr = self.cap.read() if self.cap else (False, None)
                if ok and fr is not None:
                    frames.append(fr)
            if not frames:
                frames = [self.video.frame_bgr.copy()]
            # pick best frame by quality: sharpness high, glare low
            best = frames[0]
            best_score = -1e18
            prevg = None
            for fr in frames:
                try:
                    roi_bgr_tmp = crop_roi(fr, self.roi)
                    live, g = compute_live_metrics(
                        roi_bgr_tmp, prevg,
                        min_brightness=self.cfg.min_brightness,
                        max_brightness=self.cfg.max_brightness,
                        min_sharpness=self.cfg.min_sharpness,
                    )
                    prevg = g
                    # heuristic score: prefer sharpness, penalize glare + instability
                    score = (live.sharpness * 1.0) - (live.glare_ratio * 800.0) - (live.stability * 25.0)
                    if score > best_score:
                        best_score = score
                        best = fr
                except Exception:
                    continue
            frame = best.copy()
        roi_bgr = crop_roi(frame, self.roi)

        # --- normalize CLAHE config safely (handles int/list/tuple/None) ---
        grid = getattr(self.cfg, "clahe_grid", (8, 8))
        if isinstance(grid, int):
            grid = (grid, grid)
        elif isinstance(grid, (list, tuple)) and len(grid) == 2:
            grid = (int(grid[0]), int(grid[1]))
        else:
            grid = (8, 8)

        clip = getattr(self.cfg, "clahe_clip", 2.0)
        try:
            clip = float(clip)
        except Exception:
            clip = 2.0

        processed = preprocess_bgr(roi_bgr, clahe_clip=clip, clahe_grid=grid)

        # --- required outputs ---
        gray = processed.get("gray")
        if gray is None:
            self.toast.show_toast("Preprocess failed: missing gray output.")
            return

        q = run_quality_checks(gray, self.cfg)

        # --- edges_closed is optional; fallback to edges ---
        edges_for_measure = processed.get("edges_closed", None)
        if edges_for_measure is None:
            edges_for_measure = processed.get("edges", None)
        if edges_for_measure is None:
            self.toast.show_toast("No edges found; check preprocess.")
            return

        m = groove_visibility_score(edges_for_measure)
        verdict = pass_fail_from_score(m["score"])

        # chip state
        if q["ok"] and verdict.upper().startswith("PASS"):
            self.chip.set_state("ok", "OK")
        elif not q["ok"] and verdict.upper().startswith("PASS"):
            self.chip.set_state("warn", "WARN")
        else:
            self.chip.set_state("fail", "FAIL")

        meta = {
            "camera_index": self.cam_index,
            "roi": self.roi,
            "quality": q,
            "measure": m,
            "verdict": verdict,
            "session": {
                "vehicle_id": self.in_vehicle.text().strip() or None,
                "tire_position": self.in_tirepos.currentText(),
                "operator": self.in_operator.text().strip() or None,
                "notes": self.in_notes.text().strip() or None,
            },
            "calibration": self.calib,
            "config": asdict(self.cfg),
        }

        ts, img_path, meta_path = save_capture(self.cfg, frame, meta)
        out_paths = save_processed(self.cfg, ts, processed)

        insert_result(self.cfg, {
            "ts": ts,
            "image_path": str(img_path),
            "roi_x": int(self.roi["x"]), "roi_y": int(self.roi["y"]),
            "roi_w": int(self.roi["w"]), "roi_h": int(self.roi["h"]),
            "brightness": float(q["metrics"]["brightness"]),
            "glare_ratio": float(q["metrics"]["glare_ratio"]),
            "sharpness": float(q["metrics"]["sharpness"]),
            "edge_density": float(m["edge_density"]),
            "continuity": float(m["continuity"]),
            "score": float(m["score"]),
            "verdict": verdict,
            "notes": "; ".join(q["reasons"]) if not q["ok"] else "",
            "vehicle_id": self.in_vehicle.text().strip() or None,
            "tire_position": self.in_tirepos.currentText(),
            "operator": self.in_operator.text().strip() or None,
            "session_notes": self.in_notes.text().strip() or None,
            "mm_per_px": float(self.calib["mm_per_px"]) if self.calib.get("mm_per_px") else None
        })

        # clean readable result
        self.result.clear()
        self.result.append(f"<h3 style='margin:0;'>Verdict: {verdict}</h3>")
        self.result.append(f"<b>Score:</b> {m['score']:.4f}")
        self.result.append(f"<b>Quality:</b> {'OK' if q['ok'] else 'FAIL'}")
        if q["reasons"]:
            self.result.append("<b>Issues:</b>")
            for r in q["reasons"]:
                self.result.append(f"• {r}")
        if self.calib.get("mm_per_px"):
            self.result.append(f"<b>Scale:</b> {self.calib['mm_per_px']:.6f} mm/px")

        self.toast.show_toast(f"Saved scan {ts}")
        self._refresh_history()

    def _on_auto_trigger_changed(self, text):
        self.auto_trigger_enabled = (text.lower() == "on")
        self.stable_ok_frames = 0
        self.toast.show_toast("Auto-trigger enabled" if self.auto_trigger_enabled else "Auto-trigger disabled")

    def export_csv_action(self):
        p = export_csv(self.cfg)
        self.toast.show_toast("CSV exported ✔")
        QMessageBox.information(self, "Export", f"Exported CSV:\n{p}")

    # ---------- History ----------
    def _refresh_history(self):
        if not self.adv_dock.isVisible():
            return
        self.history.clear()
        items = list_results(self.cfg, limit=40)
        for it in items:
            self.history.addItem(f"{it['ts']} | {it['verdict']} | {it['score']:.4f}")

    def on_history_select(self):
        if not self.history.selectedItems():
            return
        ts = self.history.selectedItems()[0].text().split("|")[0].strip()
        row = get_result_by_ts(self.cfg, ts)
        if not row:
            return
        self.result.clear()
        self.result.append(f"<b>Loaded:</b> {ts}")
        self.result.append(f"Verdict: {row['verdict']} | Score: {row['score']:.4f}")
        if row.get("mm_per_px"):
            self.result.append(f"Scale: {row['mm_per_px']:.6f} mm/px")
        if row.get("session_notes"):
            self.result.append(f"Notes: {row['session_notes']}")

    # ---------- Advanced ----------
    def toggle_advanced(self):
        vis = not self.adv_dock.isVisible()
        self.adv_dock.setVisible(vis)
        if vis:
            self._refresh_history()

def run_app(cfg):
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    w.show()
    sys.exit(app.exec())