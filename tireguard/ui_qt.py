# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import time
from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize, QEvent
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction, QFont, QDoubleValidator
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
from .calibration import (
    load_calibration, save_calibration, compute_scale_from_two_points, score_to_depth_mm
)
from .live_metrics import compute_live_metrics
from .storage import (
    init_db, save_capture, save_processed, insert_result, export_csv,
    list_results, get_result_by_ts, find_processed_images, insert_validation_result,
    export_validation_summary
)

# ---------- Constants ----------
VALIDATION_THRESHOLDS = {
    "max_percent_diff": 5.0,      # ≤5% difference acceptable
    "max_abs_error_mm": 0.5,      # ≤0.5mm error acceptable
    "max_processing_time_s": 5.0, # ≤5s processing time acceptable
}

PSI_DEFAULT_RECOMMENDED = 32.0
PSI_GOOD_DELTA = 1.5
PSI_WARN_DELTA = 3.0

# ---------- Helper Functions ----------
def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert OpenCV BGR image to QImage for display."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

def clamp_roi(roi: Dict[str, float], frame_shape: Tuple) -> Dict[str, int]:
    """Ensure ROI stays within frame boundaries."""
    fh, fw = frame_shape[:2]
    x = max(0, min(fw-1, int(roi["x"])))
    y = max(0, min(fh-1, int(roi["y"])))
    w = max(1, min(fw-x, int(roi["w"])))
    h = max(1, min(fh-y, int(roi["h"])))
    return {"x": x, "y": y, "w": w, "h": h}

# ---------- Toast Notification Widget ----------
class Toast(QWidget):
    """Non-blocking toast notification with animation."""
    
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
        if parent:
            parent.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.parent() and event.type() == QEvent.Resize:
            self._update_position()
        return False

    def _update_position(self):
        """Position toast at top-center of parent."""
        parent = self.parentWidget()
        if not parent or not parent.isVisible():
            return
        w = min(420, max(220, self.width()))
        h = self.height()
        margin = 20
        x = (parent.width() - w) // 2
        y = margin
        self.setGeometry(x, y, w, h)

    def show_toast(self, text: str, ms: int = 1800):
        """Display toast for specified duration."""
        self.label.setText(text)
        self.adjustSize()
        self._update_position()
        self.show()
        self.anim.stop()
        self.anim.setDuration(220)
        self.anim.setStartValue(self.geometry())
        self.anim.setEndValue(self.geometry())
        self.anim.start()
        QTimer.singleShot(ms, self._hide)

    def _hide(self):
        self.hide()

# ---------- Status Chip Widget ----------
class StatusChip(QLabel):
    """Status indicator with color-coded states."""
    
    STATE_STYLES = {
        "ok": "background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.5); color: #22c55e;",        # GREEN
        "warn": "background: rgba(251,146,60,0.15); border: 1px solid rgba(251,146,60,0.5); color: #fb923c;",    # ORANGE
        "fail": "background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.5); color: #ef4444;",      # RED
        "ready": "background: rgba(120,220,255,0.10); border: 1px solid rgba(120,220,255,0.35); color: #9bdcff;", # CYAN
    }
    
    def __init__(self, text: str = "READY"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(28)
        self.setStyleSheet("border-radius: 14px; padding: 4px 10px; font-weight: 800; letter-spacing: 0.5px;")
        self.set_state("ready")

    def set_state(self, state: str, text: Optional[str] = None):
        """Update state and optionally change label text."""
        if text is not None:
            self.setText(text)
        style = self.STATE_STYLES.get(state, self.STATE_STYLES["ready"])
        self.setStyleSheet("border-radius: 14px; padding: 4px 10px; font-weight: 800; letter-spacing: 0.5px;" + style)

# ---------- Video Display Widget ----------
class VideoWidget(QLabel):
    """Video preview with ROI/calibration overlay."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(820, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0b0f14; border-radius:18px; border: 1px solid rgba(120,220,255,0.18);")
        self.frame_bgr: Optional[np.ndarray] = None
        self.roi: Optional[Dict] = None
        self.roi_mode = False
        self.dragging = False
        self.drag_start: Optional[Tuple] = None
        self.drag_end: Optional[Tuple] = None
        self.calib_mode = False
        self.calib_points: list = []
        self._resize_mode = False

    def set_modes(self, roi_mode: bool = False, calib_mode: bool = False):
        """Set interaction mode (ROI editing or calibration)."""
        self.roi_mode = roi_mode
        self.calib_mode = calib_mode
        if not calib_mode:
            self.calib_points = []
            self.dragging = False
            self.drag_start = None
            self.drag_end = None
            self._resize_mode = False
        self.update()

    def set_frame(self, frame_bgr: np.ndarray, roi: Optional[Dict] = None):
        """Update displayed frame and optionally ROI."""
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
        self._resize_mode = bool(e.modifiers() & Qt.ShiftModifier)
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
        w = self.window()
        if hasattr(w, 'on_roi_drag_finished'):
            w.on_roi_drag_finished()
        self._resize_mode = False

    def _widget_to_frame(self, x: int, y: int) -> Tuple[int, int]:
        """Convert widget coordinates to frame coordinates."""
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
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pix)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate overlay coordinates
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

        # Draw saved ROI
        if self.roi:
            x, y, w, h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
            x0, y0 = f2w(x, y)
            x1, y1 = f2w(x + w, y + h)
            pen = QPen(QColor("#00ffb2"), 3)
            painter.setPen(pen)
            painter.drawRoundedRect(x0, y0, x1-x0, y1-y0, 14, 14)

        # Draw ROI drag preview
        if self.roi_mode and self.drag_start and self.drag_end:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_end
            pen = QPen(QColor("#7bdcff"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1-x0)), int(abs(y1-y0)))

        # Draw calibration points
        if self.calib_mode and self.calib_points:
            pen = QPen(QColor("#7bdcff"), 3)
            painter.setPen(pen)
            for (x, y) in self.calib_points:
                painter.drawEllipse(int(x)-7, int(y)-7, 14, 14)
            if len(self.calib_points) == 2:
                p0, p1 = self.calib_points
                painter.drawLine(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
        
        painter.end()

# ---------- Main Application Window ----------
class MainWindow(QMainWindow):
    """TireGuard main UI with 4-step wizard workflow."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        init_db(cfg)
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 800)
        
        # Core state
        self.cap = None
        self.cam_index = None
        self.roi: Optional[Dict] = None
        self.calib = load_calibration(cfg.calibration_path)
        
        # Live metrics state
        self.prev_roi_gray = None
        self.live_last = None
        self.auto_trigger_enabled = False
        self.stable_ok_frames = 0
        self._metrics_last_t = 0.0
        
        # Capture safety
        self._auto_last_capture_t = 0.0
        self._auto_cooldown_s = 2.0
        self._capture_busy = False
        
        self.toast = Toast(self)
        self._apply_theme()
        self._build_ui()
        self._open_camera()
        self._refresh_history()
        
        # Start main loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(15)

    def _apply_theme(self):
        """Apply dark theme with cyan accents."""
        self.setStyleSheet("""
            QMainWindow {
                background: #070a10;
            }
            QLabel {
                color: #e8f7ff;
            }
            QPushButton {
                background: rgba(20, 28, 40, 0.9);
                color: #e8f7ff;
                border: 1px solid rgba(120, 220, 255, 0.22);
                padding: 12px;
                border-radius: 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(30, 44, 66, 0.95);
                border: 1px solid rgba(120, 220, 255, 0.35);
            }
            QPushButton:pressed {
                background: rgba(12, 18, 28, 1.0);
            }
            QLineEdit, QComboBox, QTextEdit {
                background: rgba(12, 18, 28, 0.95);
                color: #e8f7ff;
                border: 1px solid rgba(120, 220, 255, 0.22);
                border-radius: 14px;
                padding: 8px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid rgba(120, 220, 255, 0.45);
            }
            QGroupBox {
                color: #e8f7ff;
                border: 1px solid rgba(120, 220, 255, 0.18);
                border-radius: 18px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                font-weight: 800;
            }
            QListWidget {
                background: rgba(12, 18, 28, 0.95);
                border: 1px solid rgba(120, 220, 255, 0.18);
                border-radius: 14px;
                color: #e8f7ff;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-radius: 10px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: rgba(30, 60, 100, 0.8);
                border: 1px solid rgba(120, 220, 255, 0.4);
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
                background: rgba(120, 220, 255, 0.4);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QWidget#sessionContainer {
                background: rgba(12, 18, 28, 0.95);
            }
            QScrollArea > QWidget {
                background: rgba(12, 18, 28, 0.95);
            }
            QComboBox QAbstractItemView {
                background: rgba(12, 18, 28, 0.95);
                border: 1px solid rgba(120, 220, 255, 0.18);
                border-radius: 12px;
                selection-background-color: rgba(30, 60, 100, 0.8);
                selection-color: #7fffd4;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px 16px;
                color: #e8f7ff;
            }
            QComboBox QAbstractItemView::item:hover {
                background: rgba(30, 60, 100, 0.5);
            }
            QComboBox QAbstractItemView::item:selected {
                background: rgba(30, 60, 100, 0.8);
                border-left: 3px solid #00ffb2;
            }
            QComboBox QLineEdit {
                background: rgba(12, 18, 28, 0.95);
                border: none;
                padding: 0 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
        """)

    # ---------- UI Building ----------
    def _build_ui(self):
        """Construct main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Top bar
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
        
        # Main content: split video + wizard
        self.video = VideoWidget()
        self.video.setParent(self)
        self.steps = QStackedWidget()
        
        self.steps.addWidget(self._step_camera())
        self.steps.addWidget(self._step_roi())
        self.steps.addWidget(self._step_calibrate())
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
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.steps)
        right_layout.addLayout(nav)
        
        right_content.setMinimumWidth(420)
        right_content.setMaximumWidth(600)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(right_content)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.setSizes([800, 400])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        main_layout.addLayout(top)
        main_layout.addWidget(splitter)
        
        self._build_advanced_dock()
        
        act_adv = QAction("Toggle Advanced", self)
        act_adv.triggered.connect(self.toggle_advanced)
        self.menuBar().addAction(act_adv)
        self._update_nav_buttons()

    def _card(self, title_text: str, body_widget: QWidget) -> QWidget:
        """Create a styled card widget."""
        box = QGroupBox(title_text)
        v = QVBoxLayout()
        v.addWidget(body_widget)
        box.setLayout(v)
        return box

    def _step_camera(self) -> QWidget:
        """Step 1: Camera selection."""
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

    def _step_roi(self) -> QWidget:
        """Step 2: ROI (Region of Interest) selection."""
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
        self.roi_info = QLabel("ROI: not set")
        self.roi_info.setStyleSheet("color: rgba(232,247,255,0.75); font-weight:700;")
        v.addWidget(self.roi_info)
        tip = QLabel("Tip: drag a rectangle around the tire tread. Keep it tight.")
        tip.setStyleSheet("color: rgba(232,247,255,0.65);")
        v.addWidget(tip)
        v.addStretch(1)
        w.setLayout(v)
        return w

    def _step_calibrate(self) -> QWidget:
        """Step 3: Optional scale calibration."""
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

    def _step_scan(self) -> QWidget:
        """Step 4: Capture and analysis."""
        w = QWidget()
        v = QVBoxLayout()
        hint = QLabel("Step 4: Scan & Save")
        hint.setStyleSheet("color: rgba(232,247,255,0.78); font-weight:700;")
        v.addWidget(hint)
        
        sess = QWidget()
        sess.setObjectName("sessionContainer")
        f = QFormLayout()
        self.in_vehicle = QLineEdit()
        self.in_operator = QLineEdit()
        self.in_tirepos = QComboBox()
        self.in_tirepos.addItems(["FL","FR","RL","RR","SPARE"])
        self.in_notes = QLineEdit()

        self.in_psi_measured = QLineEdit()
        self.in_psi_measured.setPlaceholderText("Measured PSI (e.g., 31.5)")
        self.in_psi_recommended = QLineEdit()
        self.in_psi_recommended.setPlaceholderText("Recommended PSI (default 32.0)")

        psi_validator = QDoubleValidator(0.0, 120.0, 1)
        self.in_psi_measured.setValidator(psi_validator)
        self.in_psi_recommended.setValidator(psi_validator)

        f.addRow("Vehicle ID", self.in_vehicle)
        f.addRow("Tire", self.in_tirepos)
        f.addRow("Operator", self.in_operator)
        f.addRow("Notes", self.in_notes)
        f.addRow("Measured PSI", self.in_psi_measured)
        f.addRow("Recommended PSI", self.in_psi_recommended)
        sess.setLayout(f)
        
        sess_scroll = QScrollArea()
        sess_scroll.setWidgetResizable(True)
        sess_scroll.setFrameShape(QFrame.NoFrame)
        sess_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sess_scroll.setWidget(sess)
        
        v.addWidget(self._card("Session", sess_scroll))
        
        self.aim_text = QLabel("Aim Assist: set ROI to see live metrics")
        self.aim_text.setStyleSheet("color: rgba(232,247,255,0.75); font-weight:700;")
        self.aim_box = QTextEdit()
        self.aim_box.setReadOnly(True)
        self.aim_box.setMinimumHeight(120)
        
        aim_scroll = QScrollArea()
        aim_scroll.setWidgetResizable(True)
        aim_scroll.setFrameShape(QFrame.NoFrame)
        aim_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        aim_scroll.setWidget(self.aim_box)
        
        v.addWidget(self._card("Aim Assist", aim_scroll))
        
        opt = QWidget()
        fo = QFormLayout()
        self.sp_burst = QSpinBox()
        self.sp_burst.setRange(1, 25)
        self.sp_burst.setValue(8)
        self.sp_stable_need = QSpinBox()
        self.sp_stable_need.setRange(1, 30)
        self.sp_stable_need.setValue(8)
        self.cb_auto_trigger = QComboBox()
        self.cb_auto_trigger.addItems(["Off", "On"])
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
        
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(180)
        
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setFrameShape(QFrame.NoFrame)
        result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        result_scroll.setWidget(self.result)
        
        v.addWidget(self._card("Result", result_scroll))
        
        v.addStretch(1)
        w.setLayout(v)
        return w

    def _build_advanced_dock(self):
        """Build Advanced settings dock (removable panel)."""
        self.adv_dock = QDockWidget("Advanced Settings", self)
        self.adv_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.adv_dock)
        self.adv_dock.setVisible(False)
        self.adv_dock.setStyleSheet("""
            QDockWidget {
                background: rgba(12, 18, 28, 0.95);
                border: 1px solid rgba(120, 220, 255, 0.15);
            }
            QDockWidget::title {
                background: rgba(20, 30, 45, 0.8);
                padding: 10px 14px;
                font-weight: 700;
                color: #e8f7ff;
                border-bottom: 1px solid rgba(120, 220, 255, 0.1);
            }
        """)

        adv = QWidget()
        adv.setStyleSheet("background: rgba(15, 22, 35, 0.9);")
        v = QVBoxLayout()
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(18)

        # === Calibration Section ===
        calib_group = QGroupBox("Calibration & Depth Model")
        calib_group.setStyleSheet("""
            QGroupBox {
                color: #e8f7ff;
                border: 1px solid rgba(120, 220, 255, 0.18);
                border-radius: 16px;
                margin-top: 12px;
                padding: 16px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        calib_layout = QVBoxLayout()

        calib_form = QFormLayout()
        calib_form.setSpacing(10)
        calib_form.setLabelAlignment(Qt.AlignRight)

        self.val_slope = QLineEdit("-6.0")
        self.val_slope.setPlaceholderText("Slope (mm per score unit)")
        self.val_intercept = QLineEdit("6.0")
        self.val_intercept.setPlaceholderText("Intercept (mm)")

        validator = QDoubleValidator(-100.0, 100.0, 3)
        self.val_slope.setValidator(validator)
        self.val_intercept.setValidator(validator)

        calib_form.addRow("Slope:", self.val_slope)
        calib_form.addRow("Intercept:", self.val_intercept)

        btn_save_calib = QPushButton("Save Linear Model to Calibration")
        btn_save_calib.clicked.connect(self._save_linear_calibration)
        calib_form.addRow("", btn_save_calib)

        calib_layout.addLayout(calib_form)
        calib_group.setLayout(calib_layout)

        # === Validation Section ===
        val_group = QGroupBox("Device Validation")
        val_group.setStyleSheet(calib_group.styleSheet())
        val_layout = QVBoxLayout()
        val_layout.setSpacing(12)

        self.tire_id_input = QLineEdit()
        self.tire_id_input.setPlaceholderText("Tire ID (e.g., TEST-01)")
        self.manual_depth_input = QLineEdit()
        self.manual_depth_input.setPlaceholderText("Manual depth (mm)")
        self.manual_depth_input.setValidator(QDoubleValidator(0.0, 10.0, 2))

        btn_run_val = QPushButton("Run Validation on Current Scan")
        btn_run_val.setStyleSheet("""
            QPushButton {
                background: #8b5cf6;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton:hover { background: #7c3aed; }
        """)
        btn_run_val.clicked.connect(self.run_validation_on_current_scan)

        val_layout.addWidget(QLabel("Tire ID:"))
        val_layout.addWidget(self.tire_id_input)
        val_layout.addWidget(QLabel("Manual Depth (mm):"))
        val_layout.addWidget(self.manual_depth_input)
        val_layout.addWidget(btn_run_val)
        val_group.setLayout(val_layout)

        # === Quality Thresholds ===
        thr = QGroupBox("Quality Thresholds")
        thr.setStyleSheet(calib_group.styleSheet())
        f = QFormLayout()
        f.setSpacing(14)
        f.setLabelAlignment(Qt.AlignRight)

        self.sp_min_bright = QSpinBox()
        self.sp_min_bright.setRange(0, 255)
        self.sp_min_bright.setValue(int(self.cfg.min_brightness))
        self.sp_min_bright.setStyleSheet("padding: 8px;")
        self.sp_max_bright = QSpinBox()
        self.sp_max_bright.setRange(0, 255)
        self.sp_max_bright.setValue(int(self.cfg.max_brightness))
        self.sp_max_bright.setStyleSheet("padding: 8px;")
        self.sp_min_sharp = QSpinBox()
        self.sp_min_sharp.setRange(0, 5000)
        self.sp_min_sharp.setValue(int(self.cfg.min_sharpness))
        self.sp_min_sharp.setStyleSheet("padding: 8px;")

        btn_apply = QPushButton("Apply Thresholds")
        btn_apply.setStyleSheet("""
            QPushButton {
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.4);
                color: #9bdcff;
                padding: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(59, 130, 246, 0.25);
                border: 1px solid rgba(59, 130, 246, 0.6);
            }
        """)
        btn_apply.clicked.connect(self.apply_thresholds)

        f.addRow("Minimum Brightness:", self.sp_min_bright)
        f.addRow("Maximum Brightness:", self.sp_max_bright)
        f.addRow("Minimum Sharpness:", self.sp_min_sharp)
        f.addRow("", btn_apply)
        thr.setLayout(f)

        # === Scan History ===
        hist = QGroupBox("Scan History")
        hist.setStyleSheet(thr.styleSheet())
        hv = QVBoxLayout()
        hv.setSpacing(12)

        self.history = QListWidget()
        self.history.setStyleSheet("""
            QListWidget {
                background: rgba(15, 22, 35, 0.85);
                border: 1px solid rgba(120, 220, 255, 0.18);
                border-radius: 12px;
                color: #e8f7ff;
                padding: 4px;
            }
            QListWidget::item {
                padding: 10px 14px;
                border-radius: 10px;
                margin: 3px 0;
            }
            QListWidget::item:selected {
                background: rgba(30, 60, 100, 0.85);
                border: 1px solid rgba(120, 220, 255, 0.45);
            }
        """)
        self.history.itemSelectionChanged.connect(self.on_history_select)
        hv.addWidget(self.history)

        self.hist_preview = QLabel("Select a scan to preview processed images")
        self.hist_preview.setMinimumHeight(180)
        self.hist_preview.setAlignment(Qt.AlignCenter)
        self.hist_preview.setStyleSheet("""
            QLabel {
                background: rgba(10, 15, 25, 0.8);
                border-radius: 14px;
                border: 1px solid rgba(120, 220, 255, 0.15);
                color: rgba(232,247,255,0.7);
                font-size: 14px;
                padding: 20px;
            }
        """)
        hv.addWidget(self.hist_preview)
        hist.setLayout(hv)

        # Assemble
        v.addWidget(calib_group)
        v.addWidget(val_group)
        v.addWidget(thr)
        v.addWidget(hist)
        v.addStretch(1)
        adv.setLayout(v)
        self.adv_dock.setWidget(adv)

    # ---------- Calibration Methods ----------
    def _save_linear_calibration(self):
        """Save linear calibration model to disk."""
        try:
            slope = float(self.val_slope.text().strip())
            intercept = float(self.val_intercept.text().strip())
        except ValueError:
            self.toast.show_toast("Invalid slope/intercept values.")
            return

        if self.calib is None:
            self.calib = {}
        self.calib["score_model"] = {
            "type": "linear",
            "slope": slope,
            "intercept": intercept
        }
        save_calibration(self.cfg.calibration_path, self.calib)
        self.toast.show_toast(f"✅ Calibration saved: depth = {slope}×score + {intercept} mm")

    def _score_to_depth_mm(self, score: float) -> float:
        """Convert score → depth using calibration model or fallback."""
        # Try calibration.json model
        if self.calib and "score_model" in self.calib:
            try:
                model = self.calib["score_model"]
                if model.get("type") == "linear":
                    slope = float(model.get("slope", -6.0))
                    intercept = float(model.get("intercept", 6.0))
                    return max(0.0, slope * score + intercept)
            except Exception as e:
                print(f"[WARN] Calibration model parse error: {e}")

        # Fallback: UI inputs
        try:
            slope = float(self.val_slope.text().strip() or "-6.0")
            intercept = float(self.val_intercept.text().strip() or "6.0")
        except ValueError:
            slope, intercept = -6.0, 6.0

        return max(0.0, slope * score + intercept)

    def _run_validation_core(self) -> Tuple[float, float, str]:
        """Core validation pipeline. Returns (score, depth_mm, verdict)."""
        frame = self.video.frame_bgr.copy()
        roi_bgr = crop_roi(frame, self.roi)

        # Preprocess
        processed = preprocess_bgr(
            roi_bgr,
            clahe_clip=getattr(self.cfg, "clahe_clip", 2.0),
            clahe_grid=getattr(self.cfg, "clahe_grid", (8, 8))
        )
        gray = processed.get("gray")
        if gray is None:
            raise RuntimeError("Preprocess failed: no grayscale image")

        # Quality check
        q = run_quality_checks(gray, self.cfg)

        # Measure
        edges_for_measure = processed.get("edges_closed", processed.get("edges"))
        if edges_for_measure is None:
            raise RuntimeError("No edges found for measurement")

        m = groove_visibility_score(edges_for_measure)
        device_score = float(m["score"])
        raw_verdict = pass_fail_from_score(device_score)

        # Convert to depth
        device_depth = self._score_to_depth_mm(device_score)

        return device_score, device_depth, raw_verdict

    # ---------- Validation Methods ----------
    def run_validation_on_current_scan(self):
        """Run validation workflow with current video frame."""
        if self.video.frame_bgr is None:
            self.toast.show_toast("No camera frame available.")
            return
        if not self.roi:
            self.toast.show_toast("Please set ROI first.")
            return

        tire_id = self.tire_id_input.text().strip() or "UNKNOWN"
        try:
            manual_mm = float(self.manual_depth_input.text())
        except ValueError:
            self.toast.show_toast("Enter a valid manual depth (mm).")
            return

        t0 = time.perf_counter()
        try:
            device_score, device_depth, verdict_raw = self._run_validation_core()
        except Exception as e:
            self.toast.show_toast(f"Validation failed: {e}")
            return

        proc_s = time.perf_counter() - t0
        abs_error = abs(device_depth - manual_mm)
        percent_diff = (abs_error / manual_mm) * 100.0 if manual_mm > 0 else 0.0

        # Evaluate criteria
        pass_pct = percent_diff <= VALIDATION_THRESHOLDS["max_percent_diff"]
        pass_err = abs_error <= VALIDATION_THRESHOLDS["max_abs_error_mm"]
        pass_time = proc_s <= VALIDATION_THRESHOLDS["max_processing_time_s"]
        overall_verdict = "PASS" if (pass_pct and pass_err and pass_time) else "FAIL"

        # Save to DB
        insert_validation_result(self.cfg, {
            "ts": time.strftime("%Y%m%d_%H%M%S"),
            "tire_id": tire_id,
            "manual_depth": manual_mm,
            "device_score": device_score,
            "device_depth": device_depth,
            "percent_diff": percent_diff,
            "abs_error_mm": abs_error,
            "processing_time": proc_s,
            "verdict": overall_verdict,
            "notes": f"Raw Verdict: {verdict_raw} | Criteria: avg%≤5, err≤0.5mm, time≤5s"
        })

        # Display results
        self.result.clear()
        self.result.append(f"<h3 style='margin:0;color:{'#10b981' if overall_verdict == 'PASS' else '#ef4444'};'>Validation: {overall_verdict}</h3>")
        self.result.append(f"<b>Tire ID:</b> {tire_id}")
        self.result.append(f"<b>Manual:</b> {manual_mm:.2f} mm")
        self.result.append(f"<b>Device:</b> {device_depth:.2f} mm")
        self.result.append(f"<b>Abs Error:</b> {abs_error:.3f} mm")
        self.result.append(f"<b>% Diff:</b> {percent_diff:.2f}%")
        self.result.append(f"<b>Proc Time:</b> {proc_s:.3f} s")
        self.result.append(f"<b>Raw CV Verdict:</b> {verdict_raw}")

        # Toast with status
        color = "#10b981" if overall_verdict == "PASS" else "#ef4444"
        msg = f"{'✅' if overall_verdict == 'PASS' else '❌'} {overall_verdict} | {tire_id} | Δ={abs_error:.3f}mm ({percent_diff:.2f}%)"
        self.toast.show_toast(msg)

    def export_validation_action(self):
        """Export validation summary to CSV."""
        p = export_validation_summary(self.cfg)
        QMessageBox.information(self, "Validation Export", f"Exported: {p}")

    # ---------- Navigation & Lifecycle ----------
    def _update_nav_buttons(self):
        """Update navigation button states."""
        idx = self.steps.currentIndex()
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < self.steps.count() - 1)
        labels = ["Camera", "ROI", "Calibrate", "Scan"]
        self.btn_next.setText(("Next → " + labels[idx+1]) if idx < 3 else "Done")

    def next_step(self):
        """Move to next wizard step."""
        idx = self.steps.currentIndex()
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
        """Move to previous wizard step."""
        idx = self.steps.currentIndex()
        if idx > 0:
            self.steps.setCurrentIndex(idx - 1)
            self._update_nav_buttons()

    # ---------- Camera Management ----------
    def _open_camera(self, force_index: Optional[int] = None):
        """Open camera with specified index."""
        pref = force_index if force_index is not None else self.cfg.cam_index
        try:
            self.cap, self.cam_index = open_camera(pref, self.cfg.width, self.cfg.height, self.cfg.fps)
            self.toast.show_toast(f"✅ Camera opened (index={self.cam_index})")
            self.chip.set_state("ready", "READY")
        except Exception as e:
            self.toast.show_toast(f"❌ Camera error: {e}")
            self.chip.set_state("fail", "NO CAM")

    def reopen_camera(self):
        """Reopen camera with new settings."""
        try:
            idx = int(self.cam_idx.currentText())
            preset = self.res_combo.currentText()
            for name, w, h in RES_PRESETS:
                if name == preset:
                    self.cfg.width = w
                    self.cfg.height = h
            if self.cap:
                self.cap.release()
            self._open_camera(force_index=idx)
        except Exception as e:
            self.toast.show_toast(f"Camera reopen failed: {e}")

    def _tick(self):
        """Main application loop (15ms)."""
        if not self.cap:
            return
        
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        
        self.video.set_frame(frame, roi=self.roi)
        
        # Live metrics
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
                    
                    if hasattr(self, 'aim_box'):
                        msg = [
                            f"Brightness: {live.brightness:.1f}  (target {self.cfg.min_brightness:.0f}-{self.cfg.max_brightness:.0f})",
                            f"Sharpness:  {live.sharpness:.1f}  (min {self.cfg.min_sharpness:.0f})",
                            f"Glare:      {live.glare_ratio*100:.1f}% (lower is better)",
                            f"Stability:  {live.stability:.2f} (lower is better)",
                            "",
                            "Hint: " + ("READY TO CAPTURE" if live.ok_hint else "Adjust angle/light/hold steady")
                        ]
                        self.aim_box.setPlainText("\n".join(msg))
                    
                    # Auto-trigger
                    if self.auto_trigger_enabled and self.steps.currentIndex() == 3:
                        if live.ok_hint:
                            self.stable_ok_frames += 1
                        else:
                            self.stable_ok_frames = 0
                        need = self.sp_stable_need.value() if hasattr(self, 'sp_stable_need') else 8
                        if self.stable_ok_frames >= need:
                            self.stable_ok_frames = 0
                            now = time.monotonic()
                            if self._capture_busy:
                                return
                            if (now - self._auto_last_capture_t) < self._auto_cooldown_s:
                                return
                            self._auto_last_capture_t = now
                            self.toast.show_toast("Auto-trigger capture…")
                            self.capture_analyze()
                except Exception as e:
                    print(f"[WARN] Metrics update failed: {e}")

    # ---------- ROI Management ----------
    def _update_roi_info(self):
        """Update ROI info display."""
        if not hasattr(self, "roi_info"):
            return
        if not self.roi:
            self.roi_info.setText("ROI: not set")
            return
        w = int(self.roi["w"])
        h = int(self.roi["h"])
        msg = f"ROI: {w} × {h} px"
        mm_per_px = self.calib.get("mm_per_px") if self.calib else None
        if mm_per_px:
            msg += f"  |  {w * mm_per_px:.1f} × {h * mm_per_px:.1f} mm"
        self.roi_info.setText(msg)

    def toggle_roi_mode(self):
        """Enable ROI drag mode."""
        self.video.set_modes(roi_mode=True, calib_mode=False)
        self.toast.show_toast("ROI mode: drag a rectangle on the video")

    def clear_roi(self):
        """Clear ROI selection."""
        self.roi = None
        self.video.roi = None
        self._update_roi_info()
        self.toast.show_toast("ROI cleared")

    def auto_roi(self):
        """Auto-detect ROI."""
        if self.video.frame_bgr is None:
            return
        try:
            self.roi = clamp_roi(suggest_roi(self.video.frame_bgr), self.video.frame_bgr.shape)
            self.video.set_frame(self.video.frame_bgr, roi=self.roi)
            self._update_roi_info()
            self.toast.show_toast("✅ Auto ROI set")
        except Exception as e:
            self.toast.show_toast(f"Auto ROI failed: {e}")

    def on_roi_drag_finished(self):
        """Handle ROI drag completion."""
        if self.video.frame_bgr is None or not self.video.drag_start or not self.video.drag_end:
            return
        x0, y0 = self.video.drag_start
        x1, y1 = self.video.drag_end
        fx0, fy0 = self.video._widget_to_frame(x0, y0)
        fx1, fy1 = self.video._widget_to_frame(x1, y1)
        resize = getattr(self.video, "_resize_mode", False)
        if resize and self.roi:
            x = int(self.roi["x"])
            y = int(self.roi["y"])
            w = abs(fx1 - x)
            h = abs(fy1 - y)
        else:
            x = min(fx0, fx1)
            y = min(fy0, fy1)
            w = abs(fx1 - fx0)
            h = abs(fy1 - fy0)
        if w < 40 or h < 40:
            self.toast.show_toast("ROI too small. Try again.")
            return
        self.roi = clamp_roi({"x": x, "y": y, "w": w, "h": h}, self.video.frame_bgr.shape)
        self.video.set_modes(False, False)
        self._update_roi_info()
        self.toast.show_toast(f"✅ ROI set ({w}×{h})")

    # ---------- Calibration Management ----------
    def toggle_calibration(self):
        """Enable calibration mode."""
        self.video.set_modes(roi_mode=False, calib_mode=True)
        self.toast.show_toast("Calibration: click 2 points with known distance")

    def on_calib_clicks_changed(self):
        """Handle calibration points entered."""
        if len(self.video.calib_points) == 2:
            from PySide6.QtWidgets import QInputDialog
            mm, ok = QInputDialog.getDouble(self, "Calibration", "Enter known distance (mm):", 10.0, 0.001, 10000.0, 3)
            if not ok:
                self.video.calib_points = []
                return
            try:
                p0 = self.video._widget_to_frame(*self.video.calib_points[0])
                p1 = self.video._widget_to_frame(*self.video.calib_points[1])
                px_per_mm, mm_per_px = compute_scale_from_two_points(p0, p1, float(mm))
                self.calib = {"px_per_mm": px_per_mm, "mm_per_px": mm_per_px, "method": "two_point"}
                save_calibration(self.cfg.calibration_path, self.calib)
                self.cal_label.setText(self._calib_text())
                self.video.set_modes(False, False)
                self.toast.show_toast("✅ Calibration saved")
            except Exception as e:
                self.toast.show_toast(f"Calibration failed: {e}")
                self.video.calib_points = []

    def clear_calibration(self):
        """Clear calibration data."""
        self.calib = {"px_per_mm": None, "mm_per_px": None, "method": None}
        save_calibration(self.cfg.calibration_path, self.calib)
        self.cal_label.setText(self._calib_text())
        self.toast.show_toast("Calibration cleared")

    def _calib_text(self) -> str:
        """Generate calibration status text."""
        if self.calib and self.calib.get("mm_per_px"):
            return f"Scale: {self.calib['mm_per_px']:.6f} mm/px ({self.calib['px_per_mm']:.2f} px/mm)"
        return "Scale: not calibrated"

    # ---------- Threshold Management ----------
    def apply_thresholds(self):
        """Apply new quality thresholds."""
        self.cfg.min_brightness = float(self.sp_min_bright.value())
        self.cfg.max_brightness = float(self.sp_max_bright.value())
        self.cfg.min_sharpness = float(self.sp_min_sharp.value())
        self.toast.show_toast("✅ Thresholds applied")

    def _to_float_or_none(self, text: str) -> Optional[float]:
        try:
            t = (text or "").strip()
            if not t:
                return None
            return float(t)
        except Exception:
            return None

    def _psi_verdict(self, measured: Optional[float], recommended: Optional[float]) -> str:
        # Missing PSI is neutral for combined verdict
        if measured is None:
            return "GOOD"
        rec = recommended if (recommended is not None and recommended > 0) else PSI_DEFAULT_RECOMMENDED
        delta = abs(float(measured) - float(rec))
        if delta <= PSI_GOOD_DELTA:
            return "GOOD"
        if delta <= PSI_WARN_DELTA:
            return "WARNING"
        return "REPLACE"

    def _combine_verdicts(self, tread_verdict: str, psi_verdict: str) -> str:
        sev = {"GOOD": 0, "WARNING": 1, "REPLACE": 2}
        inv = {0: "GOOD", 1: "WARNING", 2: "REPLACE"}
        return inv[max(sev.get(tread_verdict, 2), sev.get(psi_verdict, 0))]

    # ---------- Capture & Analysis ----------
    def capture_analyze(self):
        """Full capture and analysis workflow."""
        if self._capture_busy:
            return
        self._capture_busy = True
        try:
            if self.video.frame_bgr is None:
                self.toast.show_toast("No camera frame.")
                return
            if not self.roi:
                self.toast.show_toast("Set ROI first.")
                return

            t0 = time.perf_counter()

            # Burst capture
            burst_n = self.sp_burst.value() if hasattr(self, 'sp_burst') else 1
            frames = []
            if burst_n <= 1:
                frames = [self.video.frame_bgr.copy()]
            else:
                for _ in range(burst_n):
                    ok, fr = self.cap.read() if self.cap else (False, None)
                    if ok and fr is not None:
                        frames.append(fr)
                if not frames:
                    frames = [self.video.frame_bgr.copy()]

            # Select best frame
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
                    score = (live.sharpness * 1.0) - (live.glare_ratio * 800.0) - (live.stability * 25.0)
                    if score > best_score:
                        best_score = score
                        best = fr
                except Exception:
                    continue

            frame = best.copy()
            roi_bgr = crop_roi(frame, self.roi)

            # Preprocess
            grid = getattr(self.cfg, "clahe_grid", (8, 8))
            if isinstance(grid, int):
                grid = (grid, grid)
            elif isinstance(grid, (list, tuple)) and len(grid) == 2:
                grid = (int(grid[0]), int(grid[1]))
            else:
                grid = (8, 8)

            clip = float(getattr(self.cfg, "clahe_clip", 2.0))
            processed = preprocess_bgr(roi_bgr, clahe_clip=clip, clahe_grid=grid)
            gray = processed.get("gray")
            if gray is None:
                self.toast.show_toast("Preprocess failed.")
                return

            # Quality
            q = run_quality_checks(gray, self.cfg)

            # Measure
            edges_for_measure = processed.get("edges_closed", processed.get("edges"))
            if edges_for_measure is None:
                self.toast.show_toast("No edges found.")
                return

            m = groove_visibility_score(edges_for_measure)
            tread_verdict = pass_fail_from_score(m["score"])

            psi_measured = self._to_float_or_none(self.in_psi_measured.text()) if hasattr(self, "in_psi_measured") else None
            psi_recommended = self._to_float_or_none(self.in_psi_recommended.text()) if hasattr(self, "in_psi_recommended") else None
            psi_status = self._psi_verdict(psi_measured, psi_recommended)
            verdict = self._combine_verdicts(tread_verdict, psi_status)

            # Save
            ts, img_path, _meta_path = save_capture(self.cfg, frame, {
                "camera_index": self.cam_index,
                "roi": self.roi,
                "quality": q,
                "measure": m,
                "verdict": verdict,
                "tread_verdict": tread_verdict,
                "psi": {
                    "measured": psi_measured,
                    "recommended": psi_recommended if psi_recommended is not None else PSI_DEFAULT_RECOMMENDED,
                    "status": psi_status,
                },
                "session": {
                    "vehicle_id": self.in_vehicle.text().strip() or None,
                    "tire_position": self.in_tirepos.currentText().strip() or None,
                    "operator": self.in_operator.text().strip() or None,
                    "notes": self.in_notes.text().strip() or None,
                },
                "calibration": self.calib,
            })
            save_processed(self.cfg, ts, processed)

            mm_per_px = self.calib.get("mm_per_px") if isinstance(self.calib, dict) else None
            insert_result(self.cfg, {
                "ts": ts,
                "image_path": str(img_path),
                "roi_x": int(self.roi["x"]),
                "roi_y": int(self.roi["y"]),
                "roi_w": int(self.roi["w"]),
                "roi_h": int(self.roi["h"]),
                "brightness": float(q["metrics"]["brightness"]),
                "glare_ratio": float(q["metrics"]["glare_ratio"]),
                "sharpness": float(q["metrics"]["sharpness"]),
                "edge_density": float(m["edge_density"]),
                "continuity": float(m["continuity"]),
                "score": float(m["score"]),
                "verdict": verdict,              # combined verdict
                "tread_verdict": tread_verdict,  # raw tread verdict
                "psi_measured": float(psi_measured) if psi_measured is not None else None,
                "psi_recommended": float(psi_recommended) if psi_recommended is not None else PSI_DEFAULT_RECOMMENDED,
                "psi_status": psi_status,
                "notes": "; ".join(q["reasons"]) if q.get("reasons") else "",
                "vehicle_id": self.in_vehicle.text().strip() or None,
                "tire_position": self.in_tirepos.currentText().strip() or None,
                "operator": self.in_operator.text().strip() or None,
                "session_notes": self.in_notes.text().strip() or None,
                "mm_per_px": float(mm_per_px) if mm_per_px else None,
            })

            proc_s = time.perf_counter() - t0
            self.result.clear()
            self.result.append(f"<h3 style='margin:0;'>Scan: {verdict}</h3>")
            self.result.append(f"<b>TS:</b> {ts}")
            self.result.append(f"<b>Tread Verdict:</b> {tread_verdict}")
            if psi_measured is not None:
                rec = psi_recommended if psi_recommended is not None else PSI_DEFAULT_RECOMMENDED
                self.result.append(f"<b>PSI:</b> {psi_measured:.1f} (rec {rec:.1f}) → {psi_status}")
            else:
                self.result.append("<b>PSI:</b> not provided (neutral)")
            self.result.append(f"<b>Score:</b> {float(m['score']):.6f}")
            self.result.append(f"<b>Edge Density:</b> {float(m['edge_density']):.6f}")
            self.result.append(f"<b>Continuity:</b> {float(m['continuity']):.3f}")
            self.result.append(f"<b>Processing:</b> {proc_s:.3f} s")
            if q.get("reasons"):
                self.result.append("<b>Quality Notes:</b> " + "; ".join(q["reasons"]))

            if verdict == "GOOD":
                self.chip.set_state("ok", "GOOD")
            elif verdict == "WARNING":
                self.chip.set_state("warn", "WARN")
            else:
                self.chip.set_state("fail", "REPLACE")

            self._refresh_history()
            self.toast.show_toast(f"✅ Saved {ts} | {verdict}")

        except Exception as e:
            self.toast.show_toast(f"Capture failed: {e}")
            self.chip.set_state("fail", "FAIL")
        finally:
            self._capture_busy = False

    def _on_auto_trigger_changed(self, text: str):
        """Handle auto-trigger toggle."""
        self.auto_trigger_enabled = (text.strip().lower() == "on")
        self.stable_ok_frames = 0
        self.toast.show_toast("Auto-trigger ON" if self.auto_trigger_enabled else "Auto-trigger OFF")

    def export_csv_action(self):
        """Export results to CSV."""
        try:
            p = export_csv(self.cfg)
            QMessageBox.information(self, "Export CSV", f"✅ Exported: {p}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ---------- History Management ----------
    def _refresh_history(self):
        """Refresh scan history list."""
        if not hasattr(self, "history"):
            return
        self.history.blockSignals(True)
        self.history.clear()
        try:
            items = list_results(self.cfg, limit=30)
            for it in items:
                self.history.addItem(f"{it['ts']} | {it['verdict']} | {it['score']:.4f}")
        except Exception as e:
            print(f"[WARN] History refresh failed: {e}")
        self.history.blockSignals(False)

    def on_history_select(self):
        """Display selected scan preview."""
        if not self.history.selectedItems():
            return
        ts = self.history.selectedItems()[0].text().split("|")[0].strip()
        try:
            row = get_result_by_ts(self.cfg, ts)
            if not row:
                return
            paths = find_processed_images(self.cfg, ts)
            candidate = None
            for key in ("gray", "edges_closed", "edges"):
                if key in paths:
                    candidate = paths[key]
                    break
            if candidate:
                img = cv2.imread(str(candidate))
                if img is not None:
                    qimg = bgr_to_qimage(img)
                    pix = QPixmap.fromImage(qimg).scaled(
                        self.hist_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.hist_preview.setPixmap(pix)
        except Exception as e:
            print(f"[WARN] History select failed: {e}")

    # ---------- Advanced Panel ----------
    def toggle_advanced(self):
        """Toggle advanced settings dock."""
        vis = not self.adv_dock.isVisible()
        self.adv_dock.setVisible(vis)
        if vis:
            self._refresh_history()

def run_app(cfg):
    """Launch the TireGuard application."""
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    w.show()
    sys.exit(app.exec())