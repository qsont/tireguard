import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import json
import subprocess

from .camera import open_camera
from .preprocess import preprocess_bgr, crop_roi
from .quality import run_quality_checks
from .measure import groove_visibility_score, pass_fail_from_score
from .auto_roi import suggest_roi
from .calibration import load_calibration, save_calibration, compute_scale_from_two_points, score_to_depth_mm
from .storage import (
    init_db, save_capture, save_processed, insert_result,
    export_csv, list_results, get_result_by_ts, find_processed_images,
    hard_delete_scan_by_ts, restore_scan_by_ts, soft_delete_scan_by_ts
)
from .config import APP_NAME, AppConfig, RES_PRESETS


LEGAL_MIN_DEPTH_MM = {
    "car": 1.6,
    "motorcycle": 1.0,
}

DEFAULT_WARNING_BAND_MM = {
    "car": 0.4,
    "motorcycle": 0.4,
}

def list_camera_indices(max_idx=6):
    # basic probing
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        ok = cap.isOpened()
        cap.release()
        if ok:
            found.append(i)
    return found

class TireGuardApp(tk.Tk):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.title(APP_NAME)
        self.geometry("1250x740")

        init_db(cfg)
        self.calib = load_calibration(cfg.calibration_path)

        self.cap = None
        self.cam_index = None

        self.roi = None
        self.roi_selecting = False
        self._roi_start = None
        self._roi_rect = None
        self._last_frame_bgr = None
        self._disp_w = 1
        self._disp_h = 1

        # calibration click state
        self.calib_mode = False
        self._calib_points = []  # two points in display coords
        self._last_scan_depth_mm = None
        self._last_scan_vehicle_type = "Car"

        self._build_ui()
        self._open_camera()
        self._refresh_history()
        self.after(10, self._update_frame)

    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.video_label = tk.Label(self, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.video_label.bind("<Button-1>", self._on_mouse_down)
        self.video_label.bind("<B1-Motion>", self._on_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self._on_mouse_up)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="TireGuard Controls", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.status_var = tk.StringVar(value="Status: starting...")
        ttk.Label(right, textvariable=self.status_var, wraplength=380).grid(row=1, column=0, sticky="ew", pady=(0, 6))

        # --- Camera controls ---
        cam_box = ttk.LabelFrame(right, text="Camera")
        cam_box.grid(row=2, column=0, sticky="ew", pady=6)
        cam_box.columnconfigure(1, weight=1)

        ttk.Label(cam_box, text="Index").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.cam_idx_var = tk.StringVar(value="0")
        self.cam_idx_combo = ttk.Combobox(cam_box, textvariable=self.cam_idx_var, values=["0"], state="readonly", width=10)
        self.cam_idx_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(cam_box, text="Resolution").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.res_var = tk.StringVar(value="1280x720")
        self.res_combo = ttk.Combobox(cam_box, textvariable=self.res_var, values=[x[0] for x in RES_PRESETS], state="readonly")
        self.res_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=3)

        btn_row = ttk.Frame(cam_box)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        ttk.Button(btn_row, text="Refresh + Open", command=self._refresh_and_open_camera).grid(row=0, column=0, sticky="ew", padx=(0,4))
        ttk.Button(btn_row, text="Apply Resolution", command=self._apply_resolution).grid(row=0, column=1, sticky="ew", padx=(4,0))

        # Optional v4l2 controls (if installed)
        v4l = ttk.Frame(cam_box)
        v4l.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(0,6))
        ttk.Button(v4l, text="Set Auto Exposure OFF", command=lambda: self._v4l2_try(["--set-ctrl=exposure_auto=1"])).pack(side="left", padx=2)
        ttk.Button(v4l, text="Set Exposure 200", command=lambda: self._v4l2_try(["--set-ctrl=exposure_absolute=200"])).pack(side="left", padx=2)
        ttk.Button(v4l, text="Set Brightness 128", command=lambda: self._v4l2_try(["--set-ctrl=brightness=128"])).pack(side="left", padx=2)

        # --- Session fields ---
        sess = ttk.LabelFrame(right, text="Session Info")
        sess.grid(row=3, column=0, sticky="ew", pady=6)
        sess.columnconfigure(1, weight=1)

        self.vehicle_var = tk.StringVar(value="")
        self.vehicle_type_var = tk.StringVar(value="Car")
        self.tirepos_var = tk.StringVar(value="FR")
        self.operator_var = tk.StringVar(value="")
        self.snotes_var = tk.StringVar(value="")

        self.car_warn_band_var = tk.StringVar(value=str(getattr(self.cfg, "car_warning_band_mm", DEFAULT_WARNING_BAND_MM["car"])))
        self.moto_warn_band_var = tk.StringVar(value=str(getattr(self.cfg, "motorcycle_warning_band_mm", DEFAULT_WARNING_BAND_MM["motorcycle"])))

        self.vehicle_type_var.trace_add("write", lambda *_: self._on_vehicle_type_changed())

        ttk.Label(sess, text="Vehicle Type").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Combobox(sess, textvariable=self.vehicle_type_var, values=["Car", "Motorcycle"], state="readonly").grid(row=0, column=1, sticky="ew", padx=6, pady=2)

        ttk.Label(sess, text="Vehicle ID").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sess, textvariable=self.vehicle_var).grid(row=1, column=1, sticky="ew", padx=6, pady=2)

        ttk.Label(sess, text="Tire").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self.tire_combo = ttk.Combobox(sess, textvariable=self.tirepos_var, values=["FL","FR","RL","RR","SPARE"], state="readonly")
        self.tire_combo.grid(row=2, column=1, sticky="ew", padx=6, pady=2)

        ttk.Label(sess, text="Operator").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sess, textvariable=self.operator_var).grid(row=3, column=1, sticky="ew", padx=6, pady=2)

        ttk.Label(sess, text="Notes").grid(row=4, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sess, textvariable=self.snotes_var).grid(row=4, column=1, sticky="ew", padx=6, pady=2)

        # --- ROI + Calibration ---
        actions = ttk.Frame(right)
        actions.grid(row=4, column=0, sticky="ew", pady=6)
        actions.columnconfigure(0, weight=1)

        ttk.Button(actions, text="Set ROI (drag)", command=self._toggle_roi_mode).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(actions, text="Auto ROI (suggest)", command=self._auto_roi).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(actions, text="Clear ROI", command=self._clear_roi).grid(row=2, column=0, sticky="ew", pady=2)

        ttk.Separator(right).grid(row=5, column=0, sticky="ew", pady=8)

        calib_box = ttk.LabelFrame(right, text="Calibration (Scale)")
        calib_box.grid(row=6, column=0, sticky="ew")
        calib_box.columnconfigure(1, weight=1)

        self.calib_lbl = tk.StringVar(value=self._calib_text())
        ttk.Label(calib_box, textvariable=self.calib_lbl, wraplength=360).grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(calib_box, text="Start 2-point calibration", command=self._start_calibration).grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        ttk.Button(calib_box, text="Clear calibration", command=self._clear_calibration).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Label(calib_box, text="Car warning band (mm)").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(calib_box, textvariable=self.car_warn_band_var).grid(row=2, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(calib_box, text="Motorcycle warning band (mm)").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(calib_box, textvariable=self.moto_warn_band_var).grid(row=3, column=1, sticky="ew", padx=6, pady=2)
        ttk.Button(calib_box, text="Save Depth Policy", command=self._save_depth_policy).grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=4)

        self.depth_policy_text = tk.Text(right, height=7, width=46)
        self.depth_policy_text.grid(row=7, column=0, sticky="nsew", pady=4)
        self.depth_policy_text.configure(state="disabled")
        self.depth_policy_text.tag_configure("chip_replace", background="#c0392b", foreground="white")
        self.depth_policy_text.tag_configure("chip_warning", background="#e67e22", foreground="white")
        self.depth_policy_text.tag_configure("chip_good", background="#27ae60", foreground="white")

        ttk.Separator(right).grid(row=8, column=0, sticky="ew", pady=8)

        # Capture controls
        ttk.Button(right, text="Capture + Analyze", command=self._capture_analyze).grid(row=9, column=0, sticky="ew", pady=2)
        ttk.Button(right, text="Export CSV", command=self._export_csv).grid(row=10, column=0, sticky="ew", pady=2)

        ttk.Separator(right).grid(row=11, column=0, sticky="ew", pady=8)

        ttk.Label(right, text="Processed Preview", font=("Arial", 12, "bold")).grid(row=12, column=0, sticky="w")
        prev = ttk.Frame(right)
        prev.grid(row=13, column=0, sticky="ew")
        prev.columnconfigure(0, weight=1)
        prev.columnconfigure(1, weight=1)

        self.norm_label = tk.Label(prev, text="norm", bg="#222", fg="white")
        self.norm_label.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        self.edges_label = tk.Label(prev, text="edges", bg="#222", fg="white")
        self.edges_label.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        ttk.Separator(right).grid(row=14, column=0, sticky="ew", pady=8)

        ttk.Label(right, text="Last Result", font=("Arial", 12, "bold")).grid(row=15, column=0, sticky="w")
        self.result_text = tk.Text(right, height=10, width=46)
        self.result_text.grid(row=16, column=0, sticky="nsew", pady=4)
        right.rowconfigure(16, weight=1)

        ttk.Label(right, text="History (click to load)", font=("Arial", 11, "bold")).grid(row=17, column=0, sticky="w")
        history_controls = ttk.Frame(right)
        history_controls.grid(row=18, column=0, sticky="ew", pady=(2, 0))
        history_controls.columnconfigure(0, weight=1)
        self.history_mode_var = tk.StringVar(value="Active")
        self.history_mode_combo = ttk.Combobox(
            history_controls,
            textvariable=self.history_mode_var,
            values=["Active", "Recycle Bin"],
            state="readonly",
            width=14,
        )
        self.history_mode_combo.grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.history_mode_combo.bind("<<ComboboxSelected>>", self._on_history_mode_changed)
        self.btn_hist_delete = ttk.Button(history_controls, text="Delete", command=self._delete_selected_history_scan)
        self.btn_hist_delete.grid(row=0, column=1, padx=2)
        self.btn_hist_restore = ttk.Button(history_controls, text="Restore", command=self._restore_selected_history_scan)
        self.btn_hist_restore.grid(row=0, column=2, padx=2)
        self.btn_hist_hard_delete = ttk.Button(history_controls, text="Hard Delete", command=self._hard_delete_selected_history_scan)
        self.btn_hist_hard_delete.grid(row=0, column=3, padx=2)
        self.btn_hist_refresh = ttk.Button(history_controls, text="Refresh", command=self._refresh_history)
        self.btn_hist_refresh.grid(row=0, column=4, padx=(6, 0))
        self.history = tk.Listbox(right, height=7)
        self.history.grid(row=19, column=0, sticky="nsew", pady=4)
        self.history.bind("<<ListboxSelect>>", self._on_history_select)
        right.rowconfigure(19, weight=1)

        self._load_roi()
        self._refresh_camera_list()
        self._on_vehicle_type_changed()
        self._refresh_depth_policy_info()
        self._update_history_action_buttons()

    def _calib_text(self):
        if self.calib.get("mm_per_px"):
            return f"Scale: {self.calib['mm_per_px']:.6f} mm/px  ({self.calib['px_per_mm']:.3f} px/mm)"
        return "Scale: not calibrated (click 2 points on a ruler/known object)"

    def _refresh_camera_list(self):
        cams = list_camera_indices(8)
        if not cams:
            cams = [0]
        self.cam_idx_combo["values"] = [str(x) for x in cams]
        if self.cam_idx_var.get() not in [str(x) for x in cams]:
            self.cam_idx_var.set(str(cams[0]))

    def _refresh_and_open_camera(self):
        self._refresh_camera_list()
        self._open_camera(force_index=int(self.cam_idx_var.get()))

    def _apply_resolution(self):
        label = self.res_var.get()
        for name, w, h in RES_PRESETS:
            if name == label:
                self.cfg.width = w
                self.cfg.height = h
        # reopen camera to apply
        idx = int(self.cam_idx_var.get())
        self._open_camera(force_index=idx)

    def _v4l2_try(self, args):
        try:
            idx = int(self.cam_idx_var.get())
            dev = f"/dev/video{idx}"
            subprocess.run(["v4l2-ctl", "-d", dev] + args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.status_var.set(f"Status: v4l2-ctl applied on {dev}")
        except Exception:
            self.status_var.set("Status: v4l2-ctl not available or control failed")

    def _open_camera(self, force_index=None):
        try:
            if self.cap is not None:
                self.cap.release()
            pref = force_index if force_index is not None else self.cfg.cam_index
            self.cap, self.cam_index = open_camera(pref, self.cfg.width, self.cfg.height, self.cfg.fps)
            self.status_var.set(f"Status: camera opened (index={self.cam_index})")
            self.cam_idx_var.set(str(self.cam_index))
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.destroy()

    def _load_roi(self):
        try:
            if self.cfg.roi_path.exists():
                self.roi = json.loads(self.cfg.roi_path.read_text())
        except Exception:
            self.roi = None

    def _save_roi(self):
        if self.roi:
            self.cfg.roi_path.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.roi_path.write_text(json.dumps(self.roi, indent=2))

    def _toggle_roi_mode(self):
        self.roi_selecting = not self.roi_selecting
        self.calib_mode = False
        self._calib_points = []
        if self.roi_selecting:
            self.status_var.set("Status: ROI mode ON — drag on video")
        else:
            self.status_var.set("Status: ROI mode OFF")

    def _auto_roi(self):
        if self._last_frame_bgr is None:
            return
        self.roi = suggest_roi(self._last_frame_bgr)
        self._save_roi()
        self.status_var.set(f"Status: Auto ROI set (x={self.roi['x']}, y={self.roi['y']}, w={self.roi['w']}, h={self.roi['h']})")

    def _clear_roi(self):
        self.roi = None
        self._roi_start = None
        self._roi_rect = None
        try:
            if self.cfg.roi_path.exists():
                self.cfg.roi_path.unlink()
        except Exception:
            pass
        self.status_var.set("Status: ROI cleared")

    # --- Calibration controls ---
    def _start_calibration(self):
        self.calib_mode = True
        self.roi_selecting = False
        self._calib_points = []
        self.status_var.set("Status: Calibration ON — click 2 points (known distance) on the video")

    def _clear_calibration(self):
        self.calib = {"px_per_mm": None, "mm_per_px": None, "method": None}
        save_calibration(self.cfg.calibration_path, self.calib)
        self.calib_lbl.set(self._calib_text())
        self.status_var.set("Status: calibration cleared")

    def _save_depth_policy(self):
        try:
            self.cfg.car_warning_band_mm = max(0.0, float(self.car_warn_band_var.get().strip()))
            self.cfg.motorcycle_warning_band_mm = max(0.0, float(self.moto_warn_band_var.get().strip()))
            self.cfg.save_runtime_settings()
            self._refresh_depth_policy_info()
            self.status_var.set("Status: depth policy saved")
        except Exception as e:
            self.status_var.set(f"Status: depth policy save failed ({e})")

    def _on_vehicle_type_changed(self):
        vt = (self.vehicle_type_var.get() or "Car").strip().lower()
        if vt == "motorcycle":
            values = ["F (Front)", "R (Rear)"]
            if self.tirepos_var.get() not in values:
                self.tirepos_var.set(values[0])
        else:
            values = ["FL", "FR", "RL", "RR", "SPARE"]
            if self.tirepos_var.get() not in values:
                self.tirepos_var.set(values[0])
        if hasattr(self, "tire_combo"):
            self.tire_combo["values"] = values
        self._refresh_depth_policy_info()

    def _legal_min_depth_mm(self, vehicle_type: str) -> float:
        vt = (vehicle_type or "car").strip().lower()
        if vt == "motorcycle":
            return float(getattr(self.cfg, "motorcycle_legal_min_depth_mm", LEGAL_MIN_DEPTH_MM["motorcycle"]))
        return float(getattr(self.cfg, "car_legal_min_depth_mm", LEGAL_MIN_DEPTH_MM["car"]))

    def _warning_band_mm(self, vehicle_type: str) -> float:
        vt = (vehicle_type or "car").strip().lower()
        try:
            self.cfg.car_warning_band_mm = max(0.0, float(self.car_warn_band_var.get().strip()))
            self.cfg.motorcycle_warning_band_mm = max(0.0, float(self.moto_warn_band_var.get().strip()))
        except Exception:
            pass
        if vt == "motorcycle":
            return float(getattr(self.cfg, "motorcycle_warning_band_mm", DEFAULT_WARNING_BAND_MM["motorcycle"]))
        return float(getattr(self.cfg, "car_warning_band_mm", DEFAULT_WARNING_BAND_MM["car"]))

    def _tread_verdict_from_depth(self, depth_mm: float, vehicle_type: str) -> str:
        min_mm = self._legal_min_depth_mm(vehicle_type)
        warn_band = max(0.0, self._warning_band_mm(vehicle_type))
        if depth_mm < min_mm:
            return "REPLACE"
        if depth_mm < (min_mm + warn_band):
            return "WARNING"
        return "GOOD"

    def _refresh_depth_policy_info(self):
        if not hasattr(self, "depth_policy_text"):
            return
        vt = (self.vehicle_type_var.get() or "Car").strip()
        min_mm = self._legal_min_depth_mm(vt)
        warn_band = self._warning_band_mm(vt)
        good_threshold = min_mm + warn_band

        self.depth_policy_text.configure(state="normal")
        self.depth_policy_text.delete("1.0", tk.END)

        if vt.lower() == "motorcycle":
            self.depth_policy_text.insert(tk.END, "Industry Ref: Legal min 1.0 mm (jurisdiction-dependent)\n")
            guidance = "New: ~4-7 mm   Near-limit: ~1.0-1.5 mm"
        else:
            self.depth_policy_text.insert(tk.END, "Industry Ref: DOT/US/EU car legal minimum: 1.6 mm\n")
            guidance = "New: ~7-8 mm   Good: ~4-6 mm   Worn: ~2-4 mm"

        self.depth_policy_text.insert(tk.END,
            f"Policy: min {min_mm:.1f} mm, band +{warn_band:.1f} mm\n")
        self.depth_policy_text.insert(tk.END, " ")
        self.depth_policy_text.insert(tk.END, "REPLACE", "chip_replace")
        self.depth_policy_text.insert(tk.END, f" <{min_mm:.1f}   ")
        self.depth_policy_text.insert(tk.END, "WARNING", "chip_warning")
        self.depth_policy_text.insert(tk.END, f" {min_mm:.1f}-{good_threshold:.1f}   ")
        self.depth_policy_text.insert(tk.END, "GOOD", "chip_good")
        self.depth_policy_text.insert(tk.END, f" \u2265{good_threshold:.1f} mm\n")
        self.depth_policy_text.insert(tk.END, guidance + "\n")

        if self._last_scan_depth_mm is not None:
            last_vt = self._last_scan_vehicle_type or vt
            last_v = self._tread_verdict_from_depth(self._last_scan_depth_mm, last_vt)
            tag = {"REPLACE": "chip_replace", "WARNING": "chip_warning", "GOOD": "chip_good"}.get(last_v, "chip_good")
            self.depth_policy_text.insert(tk.END, "Last Scan: ")
            self.depth_policy_text.insert(tk.END, last_v, tag)
            self.depth_policy_text.insert(tk.END, f"  {self._last_scan_depth_mm:.3f} mm\n")

        self.depth_policy_text.configure(state="disabled")

    def _on_mouse_down(self, event):
        if self.calib_mode:
            self._calib_points.append((event.x, event.y))
            if len(self._calib_points) == 2:
                mm = simpledialog.askfloat("Calibration", "Enter known distance (mm):", minvalue=0.01)
                if mm is None:
                    self._calib_points = []
                    return
                # convert display coords -> frame coords
                p0 = self._disp_to_frame(self._calib_points[0])
                p1 = self._disp_to_frame(self._calib_points[1])
                px_per_mm, mm_per_px = compute_scale_from_two_points(p0, p1, float(mm))
                self.calib = {"px_per_mm": px_per_mm, "mm_per_px": mm_per_px, "method": "two_point"}
                save_calibration(self.cfg.calibration_path, self.calib)
                self.calib_lbl.set(self._calib_text())
                self.status_var.set("Status: calibration saved")
                self.calib_mode = False
                self._calib_points = []
            return

        if not self.roi_selecting:
            return
        self._roi_start = (event.x, event.y)
        self._roi_rect = (event.x, event.y, event.x, event.y)

    def _on_mouse_drag(self, event):
        if not self.roi_selecting or not self._roi_start:
            return
        x0, y0 = self._roi_start
        self._roi_rect = (x0, y0, event.x, event.y)

    def _on_mouse_up(self, event):
        if not self.roi_selecting or not self._roi_start or self._last_frame_bgr is None:
            return

        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        self._roi_start = None

        fx0, fy0 = self._disp_to_frame((x0, y0))
        fx1, fy1 = self._disp_to_frame((x1, y1))

        x = min(fx0, fx1); y = min(fy0, fy1)
        w = abs(fx1 - fx0); h = abs(fy1 - fy0)

        if w < 40 or h < 40:
            self.status_var.set("Status: ROI too small — try again")
            return

        self.roi = {"x": x, "y": y, "w": w, "h": h}
        self._save_roi()
        self.status_var.set(f"Status: ROI saved (x={x}, y={y}, w={w}, h={h})")
        self.roi_selecting = False

    def _disp_to_frame(self, p):
        if self._last_frame_bgr is None:
            return (0, 0)
        frame_h, frame_w = self._last_frame_bgr.shape[:2]
        x = int(p[0] * frame_w / max(1, self._disp_w))
        y = int(p[1] * frame_h / max(1, self._disp_h))
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        return (x, y)

    def _draw_roi_on_frame(self, frame_bgr):
        if self.roi:
            x, y, w, h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame_bgr

    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status_var.set("Status: camera frame failed")
            self.after(50, self._update_frame)
            return

        self._last_frame_bgr = frame.copy()
        preview = self._draw_roi_on_frame(frame.copy())

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        label_w = max(1, self.video_label.winfo_width())
        label_h = max(1, self.video_label.winfo_height())
        img = img.resize((label_w, label_h))
        self._disp_w, self._disp_h = label_w, label_h

        # calibration point markers (display-space)
        if self.calib_mode and self._calib_points:
            import PIL.ImageDraw as ImageDraw
            d = ImageDraw.Draw(img)
            for (x, y) in self._calib_points:
                d.ellipse([x-6, y-6, x+6, y+6], outline="yellow", width=3)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(10, self._update_frame)

    def _set_thumbnail(self, tk_label, img_path, max_size=(170, 130)):
        try:
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im.thumbnail(max_size)
            tk_img = ImageTk.PhotoImage(im)
            tk_label.configure(image=tk_img, text="")
            tk_label.image = tk_img
        except Exception:
            pass

    def _capture_analyze(self):
        if self._last_frame_bgr is None:
            return
        if not self.roi:
            messagebox.showwarning("ROI Required", "Please set ROI (drag) or use Auto ROI.")
            return

        frame = self._last_frame_bgr.copy()
        roi_bgr = crop_roi(frame, self.roi)

        processed = preprocess_bgr(roi_bgr, clahe_clip=self.cfg.clahe_clip, clahe_grid=self.cfg.clahe_grid)
        q = run_quality_checks(processed["gray"], self.cfg)
        m = groove_visibility_score(processed["edges_closed"])
        raw_score_verdict = pass_fail_from_score(m["score"])
        device_depth = score_to_depth_mm(float(m["score"]), calib=self.calib)
        vehicle_type = self.vehicle_type_var.get().strip() or "Car"
        tread_verdict = self._tread_verdict_from_depth(device_depth, vehicle_type)
        verdict = tread_verdict

        mm_per_px = self.calib.get("mm_per_px")

        meta = {
            "camera_index": self.cam_index,
            "roi": self.roi,
            "quality": q,
            "measure": m,
            "verdict": verdict,
            "tread_verdict": tread_verdict,
            "session": {
                "vehicle_type": vehicle_type,
                "vehicle_id": self.vehicle_var.get().strip() or None,
                "tire_position": self.tirepos_var.get().strip() or None,
                "operator": self.operator_var.get().strip() or None,
                "notes": self.snotes_var.get().strip() or None,
            },
            "calibration": self.calib
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
            "device_depth_mm": float(device_depth),
            "raw_score_verdict": raw_score_verdict,
            "verdict": verdict,
            "tread_verdict": tread_verdict,
            "notes": "; ".join(q["reasons"]) if not q["ok"] else "",
            "vehicle_type": vehicle_type,
            "vehicle_id": self.vehicle_var.get().strip() or None,
            "tire_position": self.tirepos_var.get().strip() or None,
            "operator": self.operator_var.get().strip() or None,
            "session_notes": self.snotes_var.get().strip() or None,
            "mm_per_px": float(mm_per_px) if mm_per_px else None
        })

        if "norm" in out_paths:
            self._set_thumbnail(self.norm_label, out_paths["norm"])
        if "edges_closed" in out_paths:
            self._set_thumbnail(self.edges_label, out_paths["edges_closed"])

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"TS: {ts}\nSaved image: {img_path}\nSaved meta: {meta_path}\n\n")
        self.result_text.insert(tk.END, f"Session: vehicle={self.vehicle_var.get()} tire={self.tirepos_var.get()} operator={self.operator_var.get()}\n")
        if mm_per_px:
            self.result_text.insert(tk.END, f"Calibration: {mm_per_px:.6f} mm/px\n")
        else:
            self.result_text.insert(tk.END, "Calibration: not set\n")
        self.result_text.insert(tk.END, f"\nQUALITY: {'OK' if q['ok'] else 'FAIL'}\n")
        if q["reasons"]:
            for r in q["reasons"]:
                self.result_text.insert(tk.END, f" - {r}\n")
        self.result_text.insert(tk.END, f"\nScore: {m['score']:.4f}\n")
        self.result_text.insert(tk.END, f"Depth (calibrated): {device_depth:.3f} mm\n")
        self.result_text.insert(tk.END, f"Raw Score Verdict: {raw_score_verdict}\n")
        self.result_text.insert(tk.END, f"Tread Verdict: {tread_verdict}\nVERDICT: {verdict}\n")

        self._last_scan_depth_mm = device_depth
        self._last_scan_vehicle_type = vehicle_type
        self._refresh_depth_policy_info()
        self.status_var.set(f"Status: captured + analyzed ({verdict})")
        self._refresh_history()

    def _refresh_history(self):
        self.history.delete(0, tk.END)
        only_deleted = getattr(self, "history_mode_var", None) is not None and self.history_mode_var.get() == "Recycle Bin"
        items = list_results(self.cfg, limit=30, only_deleted=only_deleted)
        for it in items:
            self.history.insert(tk.END, f"{it['ts']} | {it['verdict']} | {it['score']:.4f}")
        self._update_history_action_buttons()

    def _on_history_select(self, event):
        if not self.history.curselection():
            self._update_history_action_buttons()
            return
        line = self.history.get(self.history.curselection()[0])
        ts = line.split("|")[0].strip()
        only_deleted = getattr(self, "history_mode_var", None) is not None and self.history_mode_var.get() == "Recycle Bin"
        row = get_result_by_ts(self.cfg, ts, include_deleted=only_deleted)
        if not row:
            self._update_history_action_buttons()
            return

        imgs = find_processed_images(self.cfg, ts)
        if "norm" in imgs:
            self._set_thumbnail(self.norm_label, imgs["norm"])
        if "edges_closed" in imgs:
            self._set_thumbnail(self.edges_label, imgs["edges_closed"])

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"TS: {row['ts']}\nImage: {row['image_path']}\n")
        self.result_text.insert(tk.END, f"Session: vehicle={row.get('vehicle_id')} tire={row.get('tire_position')} operator={row.get('operator')}\n")
        if row.get("mm_per_px"):
            self.result_text.insert(tk.END, f"Calibration: {row['mm_per_px']:.6f} mm/px\n")
        if row.get("device_depth_mm") is not None:
            self.result_text.insert(tk.END, f"Depth (calibrated): {float(row['device_depth_mm']):.3f} mm\n")
        if row.get("raw_score_verdict"):
            self.result_text.insert(tk.END, f"Raw Score Verdict: {row['raw_score_verdict']}\n")
        self.result_text.insert(tk.END, f"\nScore: {row['score']:.4f}\nVERDICT: {row['verdict']}\n")
        if row.get("session_notes"):
            self.result_text.insert(tk.END, f"\nNotes: {row['session_notes']}\n")
        self._update_history_action_buttons()

    def _selected_history_ts(self):
        if not self.history.curselection():
            return None
        line = self.history.get(self.history.curselection()[0])
        return line.split("|")[0].strip()

    def _on_history_mode_changed(self, _event=None):
        self.result_text.delete("1.0", tk.END)
        self._refresh_history()

    def _update_history_action_buttons(self):
        in_recycle = getattr(self, "history_mode_var", None) is not None and self.history_mode_var.get() == "Recycle Bin"
        has_sel = self._selected_history_ts() is not None
        if hasattr(self, "btn_hist_delete"):
            self.btn_hist_delete.state(["!disabled"] if (has_sel and not in_recycle) else ["disabled"])
        if hasattr(self, "btn_hist_restore"):
            self.btn_hist_restore.state(["!disabled"] if (has_sel and in_recycle) else ["disabled"])
        if hasattr(self, "btn_hist_hard_delete"):
            self.btn_hist_hard_delete.state(["!disabled"] if (has_sel and in_recycle) else ["disabled"])

    def _delete_selected_history_scan(self):
        ts = self._selected_history_ts()
        if not ts:
            return
        if not messagebox.askyesno("Delete Scan", f"Move scan {ts} to recycle bin?"):
            return
        out = soft_delete_scan_by_ts(self.cfg, ts)
        if out.get("deleted", 0):
            self.result_text.delete("1.0", tk.END)
            self.status_var.set(f"Status: moved {ts} to recycle bin")
            self._refresh_history()

    def _restore_selected_history_scan(self):
        ts = self._selected_history_ts()
        if not ts:
            return
        out = restore_scan_by_ts(self.cfg, ts)
        if out.get("restored", 0):
            self.result_text.delete("1.0", tk.END)
            self.status_var.set(f"Status: restored {ts}")
            self._refresh_history()

    def _hard_delete_selected_history_scan(self):
        ts = self._selected_history_ts()
        if not ts:
            return
        if not messagebox.askyesno("Hard Delete Scan", f"Permanently delete scan {ts} and its files?"):
            return
        out = hard_delete_scan_by_ts(self.cfg, ts, delete_files=True)
        if out.get("deleted", 0):
            self.result_text.delete("1.0", tk.END)
            self.status_var.set(f"Status: hard deleted {ts}")
            self._refresh_history()

    def _export_csv(self):
        p = export_csv(self.cfg)
        messagebox.showinfo("Export", f"Exported: {p}")

def run_app(cfg):
    app = TireGuardApp(cfg)
    app.mainloop()
