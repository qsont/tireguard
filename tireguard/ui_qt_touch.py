import json
import sys
from typing import Dict, Optional, Tuple

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .auto_roi import suggest_roi
from .camera import open_camera
from .preprocess import crop_roi, preprocess_bgr
from .quality import run_quality_checks
from .measure import groove_visibility_score, pass_fail_from_score
from .storage import export_csv, init_db, insert_result, save_capture, save_processed


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

        self.setWindowTitle("TireGuard - Simple 800x480")
        self.resize(800, 480)
        self.setMinimumSize(800, 480)

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
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.status)
        root.addLayout(top)

        self.roi_info = QLabel("ROI: not set")
        self.roi_info.setStyleSheet("color:#cde9ff; font-size:14px; font-weight:700;")
        root.addWidget(self.roi_info)

        self.video = TouchVideo()
        self.video.on_roi_selected = self._on_roi_selected
        root.addWidget(self.video, 3)

        meta = QHBoxLayout()
        self.in_vehicle = QLineEdit()
        self.in_vehicle.setPlaceholderText("Vehicle ID")
        self.in_tire = QComboBox()
        self.in_tire.addItems(["FL", "FR", "RL", "RR", "SPARE"])
        meta.addWidget(self.in_vehicle, 2)
        meta.addWidget(self.in_tire, 1)
        root.addLayout(meta)

        buttons1 = QHBoxLayout()
        self.btn_roi = QPushButton("Set ROI")
        self.btn_auto_roi = QPushButton("Auto ROI")
        self.btn_clear_roi = QPushButton("Clear ROI")
        buttons1.addWidget(self.btn_roi)
        buttons1.addWidget(self.btn_auto_roi)
        buttons1.addWidget(self.btn_clear_roi)
        root.addLayout(buttons1)

        buttons2 = QHBoxLayout()
        self.btn_capture = QPushButton("Capture + Analyze")
        self.btn_export = QPushButton("Export CSV")
        buttons2.addWidget(self.btn_capture, 2)
        buttons2.addWidget(self.btn_export, 1)
        root.addLayout(buttons2)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(84)
        self.result.setMaximumHeight(110)
        root.addWidget(self.result, 1)

        self.btn_roi.clicked.connect(self._toggle_roi_mode)
        self.btn_auto_roi.clicked.connect(self._auto_roi)
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        self.btn_capture.clicked.connect(self._capture_analyze)
        self.btn_export.clicked.connect(self._export_csv)
        self._update_roi_info()

    def _set_status(self, text: str):
        self.status.setText(text)

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
            self.cap, self.cam_index = open_camera(self.cfg.cam_index, self.cfg.width, self.cfg.height, self.cfg.fps)
            self._set_status(f"Camera {self.cam_index} connected")
        except Exception as exc:
            QMessageBox.critical(self, "Camera Error", str(exc))
            self.close()

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
        verdict = pass_fail_from_score(float(m["score"]))

        meta = {
            "camera_index": self.cam_index,
            "roi": self.roi,
            "quality": q,
            "measure": m,
            "verdict": verdict,
            "session": {
                "vehicle_id": self.in_vehicle.text().strip() or None,
                "tire_position": self.in_tire.currentText(),
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
                "notes": "; ".join(q["reasons"]) if not q["ok"] else "",
                "vehicle_id": self.in_vehicle.text().strip() or None,
                "tire_position": self.in_tire.currentText(),
            },
        )

        self.result.setPlainText(
            f"TS: {ts}\n"
            f"Vehicle: {self.in_vehicle.text().strip() or '-'}\n"
            f"Tire: {self.in_tire.currentText()}\n"
            f"Score: {float(m['score']):.4f}\n"
            f"Verdict: {verdict}\n"
            f"Brightness: {float(q['metrics']['brightness']):.2f}\n"
            f"Sharpness: {float(q['metrics']['sharpness']):.2f}\n"
        )
        self._set_status(f"Captured: {verdict}")

    def _export_csv(self):
        out = export_csv(self.cfg)
        self._set_status(f"CSV exported: {out}")
        QMessageBox.information(self, "Export CSV", f"Exported: {out}")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def run_app(cfg):
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    screen = QApplication.primaryScreen()
    if screen and screen.geometry().width() <= 900 and screen.geometry().height() <= 540:
        w.showFullScreen()
    else:
        w.show()
    sys.exit(app.exec())
