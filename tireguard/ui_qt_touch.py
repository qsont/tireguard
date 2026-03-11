import json
import sys
from typing import Dict, Optional, Tuple

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDoubleValidator, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .auto_roi import suggest_roi
from .camera import open_camera
from .preprocess import crop_roi, preprocess_bgr
from .quality import run_quality_checks
from .measure import groove_visibility_score, pass_fail_from_score
from .storage import (
    export_csv,
    find_processed_images,
    get_result_by_ts,
    init_db,
    insert_result,
    list_results,
    save_capture,
    save_processed,
)


def bgr_to_qimage(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def clamp_roi(roi: Dict[str, int], frame_shape: Tuple[int, int, int]) -> Dict[str, int]:
    fh, fw = frame_shape[:2]
    x = max(0, min(fw - 1, int(roi["x"])))
    y = max(0, min(fh - 1, int(roi["y"])))
    w = max(1, min(fw - x, int(roi["w"])))
    h = max(1, min(fh - y, int(roi["h"])))
    return {"x": x, "y": y, "w": w, "h": h}


class TouchVideo(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0a1018; border: 2px solid #1b2f45; border-radius: 12px;")
        self.frame_bgr = None
        self.roi: Optional[Dict[str, int]] = None
        self.roi_mode = False
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.on_roi_selected = None

    def set_frame(self, frame_bgr, roi: Optional[Dict[str, int]] = None):
        self.frame_bgr = frame_bgr
        if roi is not None:
            self.roi = roi
        self.update()

    def set_roi_mode(self, enabled: bool):
        self.roi_mode = enabled
        if not enabled:
            self.dragging = False
            self.drag_start = None
            self.drag_end = None
        self.update()

    def _widget_to_frame(self, x: float, y: float):
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
        fx = int((x - ox) / scale)
        fy = int((y - oy) / scale)
        fx = max(0, min(fw - 1, fx))
        fy = max(0, min(fh - 1, fy))
        return fx, fy

    def mousePressEvent(self, event):
        if not self.roi_mode or self.frame_bgr is None:
            return
        self.dragging = True
        self.drag_start = (event.position().x(), event.position().y())
        self.drag_end = self.drag_start
        self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        self.drag_end = (event.position().x(), event.position().y())
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return
        self.dragging = False
        self.drag_end = (event.position().x(), event.position().y())

        if self.drag_start and self.drag_end and self.frame_bgr is not None:
            x0, y0 = self._widget_to_frame(*self.drag_start)
            x1, y1 = self._widget_to_frame(*self.drag_end)
            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            if w >= 40 and h >= 40 and callable(self.on_roi_selected):
                self.on_roi_selected({"x": x, "y": y, "w": w, "h": h})

        self.drag_start = None
        self.drag_end = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.frame_bgr is None:
            return

        qimg = bgr_to_qimage(self.frame_bgr)
        pix = QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        fh, fw = self.frame_bgr.shape[:2]
        ww = max(1, self.width())
        wh = max(1, self.height())
        scale = min(ww / fw, wh / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        ox = (ww - disp_w) // 2
        oy = (wh - disp_h) // 2

        def f2w(px, py):
            return int(ox + px * scale), int(oy + py * scale)

        if self.roi:
            x0, y0 = f2w(self.roi["x"], self.roi["y"])
            x1, y1 = f2w(self.roi["x"] + self.roi["w"], self.roi["y"] + self.roi["h"])
            painter.setPen(QPen(Qt.green, 3))
            painter.drawRect(x0, y0, x1 - x0, y1 - y0)

        if self.roi_mode and self.drag_start and self.drag_end:
            sx, sy = self.drag_start
            ex, ey = self.drag_end
            painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            painter.drawRect(int(min(sx, ex)), int(min(sy, ey)), int(abs(ex - sx)), int(abs(ey - sy)))

        painter.end()


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        init_db(cfg)

        self.cap = None
        self.cam_index = None
        self.roi: Optional[Dict[str, int]] = None
        self.psi_default_recommended = 32.0
        self.psi_good_delta = 1.5
        self.psi_warn_delta = 3.0

        self.setWindowTitle("TireGuard - Simple 800x480")
        self.setFixedSize(800, 480)

        self._load_roi()
        self._build_ui()
        self._open_camera()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    def _build_ui(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #070d14; }
            QLabel { color: #e8f3ff; }
            QPushButton {
                background: #1c344a;
                color: #ffffff;
                border: 1px solid #2b5678;
                border-radius: 10px;
                min-height: 54px;
                font-size: 18px;
                font-weight: 700;
                padding: 8px 12px;
            }
            QPushButton:hover { background: #254761; }
            QLineEdit, QComboBox, QTextEdit {
                background: #0e1a28;
                color: #e8f3ff;
                border: 1px solid #2b5678;
                border-radius: 8px;
                min-height: 40px;
                font-size: 17px;
                padding: 6px;
            }
            QScrollArea {
                border: 1px solid #1b2f45;
                border-radius: 8px;
                background: #07101b;
            }
            QScrollBar:vertical {
                width: 10px;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        top = QHBoxLayout()
        title = QLabel("TireGuard Simple")
        title.setFont(QFont("Arial", 17, QFont.Bold))
        self.status = QLabel("Ready")
        self.status.setStyleSheet("color:#9fd9ff; font-weight:700; font-size:14px;")
        self.btn_fullscreen = QPushButton("Fullscreen")
        self.btn_fullscreen.setMinimumHeight(40)
        self.btn_fullscreen.setStyleSheet(
            "QPushButton { min-height: 40px; font-size: 14px; padding: 6px 10px; border-radius: 8px; }"
        )
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.btn_fullscreen)
        top.addWidget(self.status)
        root.addLayout(top)

        body = QWidget()
        self.body_layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        body.setLayout(self.body_layout)
        root.addWidget(body, 1)

        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.roi_info = QLabel("ROI: not set")
        self.roi_info.setStyleSheet("color:#cde9ff; font-size:14px; font-weight:700;")
        left_layout.addWidget(self.roi_info)

        self.video = TouchVideo()
        self.video.on_roi_selected = self._on_roi_selected
        left_layout.addWidget(self.video, 1)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { min-height: 32px; min-width: 110px; font-size: 15px; font-weight: 700; }")
        right_layout.addWidget(self.tabs, 1)

        self.body_layout.addWidget(self.left_panel, 3)
        self.body_layout.addWidget(self.right_panel, 2)

        tab_scan = QWidget()
        scan_layout = QVBoxLayout(tab_scan)
        scan_layout.setContentsMargins(4, 4, 4, 4)
        scan_layout.setSpacing(6)

        self.scan_buttons1 = QBoxLayout(QBoxLayout.LeftToRight)
        self.btn_roi = QPushButton("Set ROI")
        self.btn_auto_roi = QPushButton("Auto ROI")
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.scan_buttons1.addWidget(self.btn_roi)
        self.scan_buttons1.addWidget(self.btn_auto_roi)
        self.scan_buttons1.addWidget(self.btn_clear_roi)
        scan_layout.addLayout(self.scan_buttons1)

        self.scan_buttons2 = QBoxLayout(QBoxLayout.LeftToRight)
        self.btn_capture = QPushButton("Capture + Analyze")
        self.btn_export = QPushButton("Export CSV")
        self.scan_buttons2.addWidget(self.btn_capture, 2)
        self.scan_buttons2.addWidget(self.btn_export, 1)
        scan_layout.addLayout(self.scan_buttons2)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(88)
        self.result.setMaximumHeight(120)
        scan_layout.addWidget(self.result)

        tab_session = QWidget()
        self.sess_form = QFormLayout(tab_session)
        self.sess_form.setContentsMargins(8, 8, 8, 8)
        self.sess_form.setSpacing(6)

        # --- Vehicle Type ---
        self.in_vehicle_type = QComboBox()
        self.in_vehicle_type.addItems(["Car", "Motorcycle"])
        self.in_vehicle_type.currentTextChanged.connect(self._on_vehicle_type_changed)

        self.in_vehicle = QLineEdit()
        self.in_vehicle.setPlaceholderText("Vehicle ID")

        self.in_tire_model = QLineEdit()
        self.in_tire_model.setPlaceholderText("e.g. Michelin Pilot Sport 4")

        self.in_tire = QComboBox()
        self.in_tire.addItems(["FL", "FR", "RL", "RR", "SPARE"])

        self.in_tire_type = QComboBox()
        self.in_tire_type.addItems([
            "— Select —",
            "All-Season Tires",
            "Summer Tires",
            "Winter/Snow Tires",
            "All-Terrain Tires",
            "Performance Tires",
            "Touring Tires",
            "Mud-Terrain Tires",
            "Run-Flat Tires",
            "Competition Tires",
            "Eco-Friendly Tires",
            "Spare Tires",
        ])

        self.in_tread_design = QComboBox()
        self.in_tread_design.addItems([
            "— Select —",
            "Symmetrical",
            "Asymmetrical",
            "Directional",
        ])

        self.in_operator = QLineEdit()
        self.in_operator.setPlaceholderText("Operator")
        self.in_notes = QLineEdit()
        self.in_notes.setPlaceholderText("Notes")
        self.in_psi_measured = QLineEdit()
        self.in_psi_measured.setPlaceholderText("Measured PSI (e.g., 31.5)")
        self.in_psi_recommended = QLineEdit()
        self.in_psi_recommended.setPlaceholderText("Recommended PSI (e.g., 32.0)")
        self.psi_status = QLabel("PSI Status: -")
        self.psi_status.setStyleSheet("color:#9fd9ff; font-size:14px; font-weight:700;")

        self.sess_form.addRow("Vehicle Type", self.in_vehicle_type)
        self.sess_form.addRow("Vehicle ID", self.in_vehicle)
        self.sess_form.addRow("Tire Model Code", self.in_tire_model)
        self.sess_form.addRow("Tire Position", self.in_tire)
        self.sess_form.addRow("Tire Type", self.in_tire_type)
        self.sess_form.addRow("Tread Design", self.in_tread_design)
        self.sess_form.addRow("Operator", self.in_operator)
        self.sess_form.addRow("PSI Measured", self.in_psi_measured)
        self.sess_form.addRow("PSI Recommended", self.in_psi_recommended)
        self.sess_form.addRow("PSI Status", self.psi_status)
        self.sess_form.addRow("Notes", self.in_notes)
        self.in_psi_measured.textChanged.connect(self._update_psi_status_label)
        self.in_psi_recommended.textChanged.connect(self._update_psi_status_label)

        tab_history = QWidget()
        history_layout = QVBoxLayout(tab_history)
        history_layout.setContentsMargins(6, 6, 6, 6)
        history_layout.setSpacing(6)
        self.history = QListWidget()
        self.history.setMinimumHeight(120)
        self.history.itemSelectionChanged.connect(self._on_history_select)
        history_layout.addWidget(self.history)
        btn_hist_refresh = QPushButton("Refresh History")
        btn_hist_refresh.clicked.connect(self._refresh_history)
        history_layout.addWidget(btn_hist_refresh)

        tab_settings = QWidget()
        set_layout = QFormLayout(tab_settings)
        set_layout.setContentsMargins(8, 8, 8, 8)
        set_layout.setSpacing(6)
        self.cam_idx = QComboBox()
        self.cam_idx.addItems([str(i) for i in range(0, 6)])
        self.cam_idx.setCurrentText("0")
        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.res_combo.setCurrentText(f"{self.cfg.width}x{self.cfg.height}")
        self.btn_open_cam = QPushButton("Open Camera")
        self.btn_open_cam.clicked.connect(self._reopen_camera)
        psi_validator = QDoubleValidator(0.0, 120.0, 1)
        self.psi_default_input = QLineEdit(f"{self.psi_default_recommended:.1f}")
        self.psi_default_input.setValidator(psi_validator)
        self.psi_good_input = QLineEdit(f"{self.psi_good_delta:.1f}")
        self.psi_good_input.setValidator(psi_validator)
        self.psi_warn_input = QLineEdit(f"{self.psi_warn_delta:.1f}")
        self.psi_warn_input.setValidator(psi_validator)
        set_layout.addRow("Camera", self.cam_idx)
        set_layout.addRow("Resolution", self.res_combo)
        set_layout.addRow("Default PSI", self.psi_default_input)
        set_layout.addRow("Good ±", self.psi_good_input)
        set_layout.addRow("Warn ±", self.psi_warn_input)
        set_layout.addRow(self.btn_open_cam)

        self.tabs.addTab(self._wrap_tab_scroll(tab_scan), "Scan")
        self.tabs.addTab(self._wrap_tab_scroll(tab_session), "Session")
        self.tabs.addTab(self._wrap_tab_scroll(tab_history), "History")
        self.tabs.addTab(self._wrap_tab_scroll(tab_settings), "Settings")

        self.btn_roi.clicked.connect(self._toggle_roi_mode)
        self.btn_auto_roi.clicked.connect(self._auto_roi)
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        self.btn_capture.clicked.connect(self._capture_analyze)
        self.btn_export.clicked.connect(self._export_csv)
        self.psi_default_input.textChanged.connect(self._update_psi_status_label)
        self.psi_good_input.textChanged.connect(self._update_psi_status_label)
        self.psi_warn_input.textChanged.connect(self._update_psi_status_label)
        self._update_roi_info()
        self._update_psi_status_label()
        self._refresh_history()
        self._sync_fullscreen_button()
        self._apply_layout_mode()

    def _wrap_tab_scroll(self, page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        # Keep all tabs navigable on 5-inch screens when content exceeds viewport.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

    def _set_status(self, text: str):
        self.status.setText(text)

    def _sync_fullscreen_button(self):
        if not hasattr(self, "btn_fullscreen"):
            return
        if self.isFullScreen():
            self.btn_fullscreen.setText("Window")
        else:
            self.btn_fullscreen.setText("Fullscreen")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.setFixedSize(800, 480)
            self.showNormal()
            self._set_status("Windowed mode")
        else:
            # Release fixed-size lock so Qt can expand to fill the screen
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
            self.showFullScreen()
            self._set_status("Fullscreen mode")
        self._sync_fullscreen_button()
        self._apply_layout_mode()

    def _apply_layout_mode(self):
        if not hasattr(self, "body_layout") or not hasattr(self, "tabs"):
            return
        if self.isFullScreen():
            # Fullscreen uses the same optimized side-by-side composition as windowed mode.
            self.body_layout.setDirection(QBoxLayout.LeftToRight)
            self.body_layout.setStretch(0, 4)
            self.body_layout.setStretch(1, 5)
            self.tabs.setTabPosition(QTabWidget.North)
            self.tabs.setStyleSheet(
                "QTabBar::tab { min-height: 30px; min-width: 92px; font-size: 14px; font-weight: 700; }"
            )
            if hasattr(self, "scan_buttons1"):
                self.scan_buttons1.setDirection(QBoxLayout.TopToBottom)
            if hasattr(self, "scan_buttons2"):
                self.scan_buttons2.setDirection(QBoxLayout.TopToBottom)
            if hasattr(self, "sess_form"):
                self.sess_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            compact_btn_style = "QPushButton { min-height: 40px; font-size: 13px; padding: 4px 8px; border-radius: 8px; }"
            for btn in (self.btn_roi, self.btn_auto_roi, self.btn_clear_roi, self.btn_capture, self.btn_export):
                btn.setStyleSheet(compact_btn_style)
            compact_input_style = (
                "QLineEdit, QComboBox { min-height: 30px; font-size: 13px; padding: 4px; border-radius: 7px; }"
            )
            for w in (
                self.in_vehicle,
                self.in_operator,
                self.in_notes,
                self.in_psi_measured,
                self.in_psi_recommended,
                self.in_tire,
            ):
                w.setStyleSheet(compact_input_style)
            self.result.setMinimumHeight(66)
            self.result.setMaximumHeight(96)
        else:
            # Windowed 800x480: camera on left, entire tab section on right.
            self.body_layout.setDirection(QBoxLayout.LeftToRight)
            self.body_layout.setStretch(0, 4)
            self.body_layout.setStretch(1, 5)
            self.tabs.setTabPosition(QTabWidget.North)
            self.tabs.setStyleSheet(
                "QTabBar::tab { min-height: 30px; min-width: 84px; font-size: 14px; font-weight: 700; }"
            )
            if hasattr(self, "scan_buttons1"):
                self.scan_buttons1.setDirection(QBoxLayout.TopToBottom)
            if hasattr(self, "scan_buttons2"):
                self.scan_buttons2.setDirection(QBoxLayout.TopToBottom)
            if hasattr(self, "sess_form"):
                self.sess_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            compact_btn_style = "QPushButton { min-height: 40px; font-size: 13px; padding: 4px 8px; border-radius: 8px; }"
            for btn in (self.btn_roi, self.btn_auto_roi, self.btn_clear_roi, self.btn_capture, self.btn_export):
                btn.setStyleSheet(compact_btn_style)
            compact_input_style = (
                "QLineEdit, QComboBox { min-height: 30px; font-size: 13px; padding: 4px; border-radius: 7px; }"
            )
            for w in (
                self.in_vehicle,
                self.in_operator,
                self.in_notes,
                self.in_psi_measured,
                self.in_psi_recommended,
                self.in_tire,
            ):
                w.setStyleSheet(compact_input_style)
            self.result.setMinimumHeight(66)
            self.result.setMaximumHeight(96)

    def _update_roi_info(self):
        if not self.roi:
            self.roi_info.setText("ROI: not set")
            return
        self.roi_info.setText(f"ROI: {self.roi['w']} x {self.roi['h']} px")

    def _load_roi(self):
        try:
            if self.cfg.roi_path.exists():
                data = json.loads(self.cfg.roi_path.read_text(encoding="utf-8"))
                if all(k in data for k in ("x", "y", "w", "h")):
                    self.roi = {k: int(data[k]) for k in ("x", "y", "w", "h")}
        except Exception:
            self.roi = None

    def _save_roi(self):
        if not self.roi:
            return
        self.cfg.roi_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.roi_path.write_text(json.dumps(self.roi, indent=2), encoding="utf-8")
        self._update_roi_info()

    def _open_camera(self):
        try:
            preferred = self.cfg.cam_index
            if hasattr(self, "cam_idx"):
                try:
                    preferred = int(self.cam_idx.currentText())
                except Exception:
                    preferred = self.cfg.cam_index
            self.cap, self.cam_index = open_camera(preferred, self.cfg.width, self.cfg.height, self.cfg.fps)
            if hasattr(self, "cam_idx"):
                self.cam_idx.setCurrentText(str(self.cam_index))
            self._set_status(f"Camera {self.cam_index} connected")
        except Exception as exc:
            QMessageBox.critical(self, "Camera Error", str(exc))
            self.close()

    def _to_float_or_none(self, text: str) -> Optional[float]:
        try:
            v = text.strip()
            return float(v) if v != "" else None
        except Exception:
            return None

    def _psi_verdict(self, measured: Optional[float], recommended: Optional[float]) -> str:
        if measured is None:
            return "NO_DATA"
        good_delta, warn_delta, target_default = self._get_psi_settings()
        target = target_default if recommended is None else float(recommended)
        delta = abs(float(measured) - target)
        is_low = float(measured) < target
        direction = "LOW" if is_low else "HIGH"
        if delta <= good_delta:
            return "GOOD"
        if delta <= warn_delta:
            return f"WARN {direction}"
        return f"CRITICAL {direction}"

    def _on_vehicle_type_changed(self, vehicle_type: str):
        """Update tire position options when vehicle type changes."""
        self.in_tire.clear()
        if vehicle_type == "Motorcycle":
            self.in_tire.addItems(["F (Front)", "R (Rear)"])
        else:
            self.in_tire.addItems(["FL", "FR", "RL", "RR", "SPARE"])

    def _get_psi_settings(self) -> Tuple[float, float, float]:
        good = self._to_float_or_none(self.psi_good_input.text()) if hasattr(self, "psi_good_input") else None
        warn = self._to_float_or_none(self.psi_warn_input.text()) if hasattr(self, "psi_warn_input") else None
        default_rec = self._to_float_or_none(self.psi_default_input.text()) if hasattr(self, "psi_default_input") else None

        good = self.psi_good_delta if good is None else max(0.1, good)
        warn = self.psi_warn_delta if warn is None else max(good + 0.1, warn)
        default_rec = self.psi_default_recommended if default_rec is None else max(0.0, default_rec)
        return good, warn, default_rec

    def _update_psi_status_label(self):
        measured = self._to_float_or_none(self.in_psi_measured.text()) if hasattr(self, "in_psi_measured") else None
        recommended = self._to_float_or_none(self.in_psi_recommended.text()) if hasattr(self, "in_psi_recommended") else None
        good_delta, warn_delta, default_rec = self._get_psi_settings()
        status = self._psi_verdict(measured, recommended)
        rec_text = recommended if recommended is not None else default_rec
        text = status if status != "NO_DATA" else f"- (target {rec_text:.1f})"
        if hasattr(self, "psi_status"):
            self.psi_status.setText(
                f"PSI Status: {text} | good ±{good_delta:.1f}, warn ±{warn_delta:.1f}"
            )

    def _reopen_camera(self):
        label = self.res_combo.currentText() if hasattr(self, "res_combo") else ""
        if "x" in label:
            try:
                w, h = label.split("x", 1)
                self.cfg.width = int(w)
                self.cfg.height = int(h)
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._open_camera()

    def _tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        self.video.set_frame(frame, roi=self.roi)

    def _toggle_roi_mode(self):
        enabled = not self.video.roi_mode
        self.video.set_roi_mode(enabled)
        self.btn_roi.setText("Stop ROI" if enabled else "Set ROI")
        self._set_status("ROI mode: drag rectangle" if enabled else "ROI mode off")

    def _on_roi_selected(self, roi: Dict[str, int]):
        if self.video.frame_bgr is None:
            return
        self.roi = clamp_roi(roi, self.video.frame_bgr.shape)
        self._save_roi()
        self.video.set_roi_mode(False)
        self.btn_roi.setText("Set ROI")
        self._set_status(f"ROI set {self.roi['w']}x{self.roi['h']}")

    def _clear_roi(self):
        self.roi = None
        self.video.roi = None
        self.video.update()
        self.btn_roi.setText("Set ROI")
        self._update_roi_info()
        self._set_status("ROI cleared")

    def _auto_roi(self):
        if self.video.frame_bgr is None:
            self._set_status("No frame yet")
            return
        try:
            self.roi = suggest_roi(self.video.frame_bgr)
            self._save_roi()
            self._set_status(f"Auto ROI set {self.roi['w']}x{self.roi['h']}")
        except Exception as exc:
            self._set_status(f"Auto ROI failed: {exc}")

    def _capture_analyze(self):
        if self.video.frame_bgr is None:
            self._set_status("No frame to capture")
            return
        if not self.roi:
            QMessageBox.warning(self, "ROI Required", "Set ROI first.")
            return

        vehicle = self.in_vehicle.text().strip()
        operator = self.in_operator.text().strip()
        psi_measured = self._to_float_or_none(self.in_psi_measured.text())
        psi_recommended = self._to_float_or_none(self.in_psi_recommended.text())
        if not vehicle or not operator or psi_measured is None or psi_recommended is None:
            QMessageBox.warning(
                self,
                "Missing Session Data",
                "Please fill Vehicle, Operator, PSI Measured, and PSI Recommended before capture.",
            )
            return

        frame = self.video.frame_bgr.copy()
        roi_bgr = crop_roi(frame, self.roi)

        processed = preprocess_bgr(
            roi_bgr,
            clahe_clip=getattr(self.cfg, "clahe_clip", 2.0),
            clahe_grid=getattr(self.cfg, "clahe_grid", (8, 8)),
        )
        q = run_quality_checks(processed["gray"], self.cfg)
        edges = processed.get("edges_closed", processed.get("edges"))
        m = groove_visibility_score(edges)
        tread_verdict = pass_fail_from_score(float(m["score"]))
        psi_status = self._psi_verdict(psi_measured, psi_recommended)
        verdict = tread_verdict
        if psi_status == "CRITICAL":
            verdict = "REPLACE"

        meta = {
            "camera_index": self.cam_index,
            "roi": self.roi,
            "quality": q,
            "measure": m,
            "verdict": verdict,
            "session": {
                "vehicle_type": self.in_vehicle_type.currentText(),
                "vehicle_id": vehicle or None,
                "tire_model_code": self.in_tire_model.text().strip() or None,
                "tire_position": self.in_tire.currentText(),
                "tire_type": self.in_tire_type.currentText() if self.in_tire_type.currentText() != "— Select —" else None,
                "tread_design": self.in_tread_design.currentText() if self.in_tread_design.currentText() != "— Select —" else None,
                "operator": operator or None,
                "psi_measured": psi_measured,
                "psi_recommended": psi_recommended,
                "psi_status": psi_status,
                "psi_good_delta": self._get_psi_settings()[0],
                "psi_warn_delta": self._get_psi_settings()[1],
                "session_notes": self.in_notes.text().strip() or None,
            },
        }

        ts, img_path, _ = save_capture(self.cfg, frame, meta)
        save_processed(self.cfg, ts, processed)

        insert_result(
            self.cfg,
            {
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
                "verdict": verdict,
                "tread_verdict": tread_verdict,
                "psi_measured": psi_measured,
                "psi_recommended": psi_recommended,
                "psi_status": psi_status,
                "notes": "; ".join(q["reasons"]) if not q["ok"] else "",
                "vehicle_type": self.in_vehicle_type.currentText(),
                "vehicle_id": vehicle or None,
                "tire_model_code": self.in_tire_model.text().strip() or None,
                "tire_position": self.in_tire.currentText(),
                "tire_type": self.in_tire_type.currentText() if self.in_tire_type.currentText() != "— Select —" else None,
                "tread_design": self.in_tread_design.currentText() if self.in_tread_design.currentText() != "— Select —" else None,
                "operator": operator or None,
                "session_notes": self.in_notes.text().strip() or None,
            },
        )

        self.result.setPlainText(
            f"TS: {ts}\n"
            f"Vehicle Type: {self.in_vehicle_type.currentText()}\n"
            f"Vehicle: {vehicle or '-'}\n"
            f"Tire Model: {self.in_tire_model.text().strip() or '-'}\n"
            f"Tire: {self.in_tire.currentText()}\n"
            f"Tire Type: {self.in_tire_type.currentText()}\n"
            f"Tread Design: {self.in_tread_design.currentText()}\n"
            f"Operator: {operator or '-'}\n"
            f"PSI Measured: {psi_measured:.1f}\n"
            f"PSI Recommended: {psi_recommended:.1f}\n"
            f"PSI Status: {psi_status}\n"
            f"Tread Verdict: {tread_verdict}\n"
            f"Score: {float(m['score']):.4f}\n"
            f"Final Verdict: {verdict}\n"
            f"Brightness: {float(q['metrics']['brightness']):.2f}\n"
            f"Sharpness: {float(q['metrics']['sharpness']):.2f}\n"
        )
        self._set_status(f"Captured: {verdict}")
        self._refresh_history()

    def _export_csv(self):
        out = export_csv(self.cfg)
        self._set_status(f"CSV exported: {out}")
        QMessageBox.information(self, "Export CSV", f"Exported: {out}")

    def _refresh_history(self):
        if not hasattr(self, "history"):
            return
        self.history.clear()
        for row in list_results(self.cfg, limit=30):
            score = float(row.get("score", 0.0)) if row.get("score") is not None else 0.0
            self.history.addItem(f"{row['ts']} | {row['verdict']} | {score:.4f}")

    def _on_history_select(self):
        if not hasattr(self, "history") or not self.history.selectedItems():
            return
        line = self.history.selectedItems()[0].text()
        ts = line.split("|")[0].strip()
        row = get_result_by_ts(self.cfg, ts)
        if not row:
            return
        imgs = find_processed_images(self.cfg, ts)
        self.result.setPlainText(
            f"TS: {row['ts']}\n"
            f"Vehicle: {row.get('vehicle_id') or '-'}\n"
            f"Tire: {row.get('tire_position') or '-'}\n"
            f"Operator: {row.get('operator') or '-'}\n"
            f"PSI Measured: {row.get('psi_measured') if row.get('psi_measured') is not None else '-'}\n"
            f"PSI Recommended: {row.get('psi_recommended') if row.get('psi_recommended') is not None else '-'}\n"
            f"PSI Status: {row.get('psi_status') or '-'}\n"
            f"Tread Verdict: {row.get('tread_verdict') or '-'}\n"
            f"Score: {float(row.get('score') or 0.0):.4f}\n"
            f"Verdict: {row.get('verdict') or '-'}\n"
            f"Processed files: {', '.join(sorted(imgs.keys())) if imgs else 'none'}\n"
        )

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def run_app(cfg):
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    screen = QApplication.primaryScreen()
    if screen and screen.geometry().width() <= 900 and screen.geometry().height() <= 540:
        # Release fixed-size lock before going fullscreen
        w.setMinimumSize(0, 0)
        w.setMaximumSize(16777215, 16777215)
        w.showFullScreen()
    else:
        w.show()  # Fixed 800x480 locked in __init__
    w._sync_fullscreen_button()
    w._apply_layout_mode()
    sys.exit(app.exec())
