import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import json

from .camera import open_camera
from .preprocess import preprocess_bgr, crop_roi
from .quality import run_quality_checks
from .measure import groove_visibility_score, pass_fail_from_score
from .auto_roi import suggest_roi
from .storage import (
    init_db, save_capture, save_processed, insert_result,
    export_csv, list_results, get_result_by_ts, find_processed_images
)
from .config import APP_NAME

class TireGuardApp(tk.Tk):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.title(APP_NAME)
        self.geometry("1200x720")

        init_db(cfg)

        self.cap = None
        self.cam_index = None

        self.roi = None
        self.roi_selecting = False
        self._roi_start = None
        self._roi_rect = None
        self._last_frame_bgr = None
        self._disp_w = 1
        self._disp_h = 1

        # for preview thumbnails
        self._thumb_norm = None
        self._thumb_edges = None

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

        ttk.Label(right, text="TireGuard Controls", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.status_var = tk.StringVar(value="Status: starting...")
        ttk.Label(right, textvariable=self.status_var, wraplength=360).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        btns = ttk.Frame(right)
        btns.grid(row=2, column=0, sticky="ew")
        btns.columnconfigure(0, weight=1)

        self.btn_set_roi = ttk.Button(btns, text="Set ROI (drag on video)", command=self._toggle_roi_mode)
        self.btn_set_roi.grid(row=0, column=0, sticky="ew", pady=3)

        self.btn_auto_roi = ttk.Button(btns, text="Auto ROI (suggest)", command=self._auto_roi)
        self.btn_auto_roi.grid(row=1, column=0, sticky="ew", pady=3)

        self.btn_clear_roi = ttk.Button(btns, text="Clear ROI", command=self._clear_roi)
        self.btn_clear_roi.grid(row=2, column=0, sticky="ew", pady=3)

        self.btn_capture = ttk.Button(btns, text="Capture + Analyze", command=self._capture_analyze)
        self.btn_capture.grid(row=3, column=0, sticky="ew", pady=(10, 3))

        self.btn_export = ttk.Button(btns, text="Export CSV", command=self._export_csv)
        self.btn_export.grid(row=4, column=0, sticky="ew", pady=3)

        ttk.Separator(right).grid(row=3, column=0, sticky="ew", pady=10)

        # Processed previews
        ttk.Label(right, text="Processed Preview", font=("Arial", 13, "bold")).grid(row=4, column=0, sticky="w")
        prev = ttk.Frame(right)
        prev.grid(row=5, column=0, sticky="ew")
        prev.columnconfigure(0, weight=1)
        prev.columnconfigure(1, weight=1)

        self.norm_label = tk.Label(prev, text="norm", bg="#222", fg="white")
        self.norm_label.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        self.edges_label = tk.Label(prev, text="edges", bg="#222", fg="white")
        self.edges_label.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        ttk.Separator(right).grid(row=6, column=0, sticky="ew", pady=10)

        # Result text + History
        ttk.Label(right, text="Last Result", font=("Arial", 13, "bold")).grid(row=7, column=0, sticky="w")
        self.result_text = tk.Text(right, height=10, width=44)
        self.result_text.grid(row=8, column=0, sticky="nsew", pady=5)
        right.rowconfigure(8, weight=1)

        ttk.Label(right, text="History (click to load)", font=("Arial", 12, "bold")).grid(row=9, column=0, sticky="w", pady=(6, 0))
        self.history = tk.Listbox(right, height=8)
        self.history.grid(row=10, column=0, sticky="nsew", pady=4)
        self.history.bind("<<ListboxSelect>>", self._on_history_select)

        self._load_roi()

    def _open_camera(self):
        try:
            self.cap, self.cam_index = open_camera(self.cfg.cam_index, self.cfg.width, self.cfg.height, self.cfg.fps)
            self.status_var.set(f"Status: camera opened (index={self.cam_index})")
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
        if self.roi_selecting:
            self.status_var.set("Status: ROI mode ON — drag on the video to select tread area")
            self.btn_set_roi.config(text="ROI mode: ON (drag now)")
        else:
            self.status_var.set("Status: ROI mode OFF")
            self.btn_set_roi.config(text="Set ROI (drag on video)")

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

    def _on_mouse_down(self, event):
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

        frame_h, frame_w = self._last_frame_bgr.shape[:2]
        disp_w = self._disp_w
        disp_h = self._disp_h

        def clamp(v, lo, hi): return max(lo, min(hi, v))
        x0 = clamp(x0, 0, disp_w-1); x1 = clamp(x1, 0, disp_w-1)
        y0 = clamp(y0, 0, disp_h-1); y1 = clamp(y1, 0, disp_h-1)

        fx0 = int(x0 * frame_w / disp_w); fx1 = int(x1 * frame_w / disp_w)
        fy0 = int(y0 * frame_h / disp_h); fy1 = int(y1 * frame_h / disp_h)

        x = min(fx0, fx1); y = min(fy0, fy1)
        w = abs(fx1 - fx0); h = abs(fy1 - fy0)

        if w < 40 or h < 40:
            self.status_var.set("Status: ROI too small — try again")
            return

        self.roi = {"x": x, "y": y, "w": w, "h": h}
        self._save_roi()
        self.status_var.set(f"Status: ROI saved (x={x}, y={y}, w={w}, h={h})")

        self.roi_selecting = False
        self.btn_set_roi.config(text="Set ROI (drag on video)")

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

        if self.roi_selecting and self._roi_rect:
            import PIL.ImageDraw as ImageDraw
            d = ImageDraw.Draw(img)
            x0, y0, x1, y1 = self._roi_rect
            d.rectangle([x0, y0, x1, y1], outline="lime", width=3)

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
            messagebox.showwarning("ROI Required", "Please set ROI (tread area) first (drag) or use Auto ROI.")
            return

        frame = self._last_frame_bgr.copy()
        roi_bgr = crop_roi(frame, self.roi)

        processed = preprocess_bgr(roi_bgr, clahe_clip=self.cfg.clahe_clip, clahe_grid=self.cfg.clahe_grid)
        q = run_quality_checks(processed["gray"], self.cfg)
        m = groove_visibility_score(processed["edges_closed"])
        verdict = pass_fail_from_score(m["score"])

        meta = {"camera_index": self.cam_index, "roi": self.roi, "quality": q, "measure": m, "verdict": verdict}
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
            "notes": "; ".join(q["reasons"]) if not q["ok"] else ""
        })

        # Update previews
        if "norm" in out_paths:
            self._set_thumbnail(self.norm_label, out_paths["norm"])
        if "edges_closed" in out_paths:
            self._set_thumbnail(self.edges_label, out_paths["edges_closed"])

        # Show result text
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"TS: {ts}\nSaved image: {img_path}\nSaved meta: {meta_path}\n\n")
        self.result_text.insert(tk.END, f"QUALITY: {'OK' if q['ok'] else 'FAIL'}\n")
        if q["reasons"]:
            self.result_text.insert(tk.END, "Reasons:\n")
            for r in q["reasons"]:
                self.result_text.insert(tk.END, f" - {r}\n")
        self.result_text.insert(tk.END, "\nMEASURE:\n")
        self.result_text.insert(tk.END, f"Edge density: {m['edge_density']:.4f}\n")
        self.result_text.insert(tk.END, f"Continuity:   {m['continuity']:.1f}\n")
        self.result_text.insert(tk.END, f"Score:        {m['score']:.4f}\n")
        self.result_text.insert(tk.END, f"\nVERDICT: {verdict}\n")

        self.status_var.set(f"Status: captured + analyzed ({verdict})")
        self._refresh_history()

    def _refresh_history(self):
        self.history.delete(0, tk.END)
        items = list_results(self.cfg, limit=30)
        for it in items:
            self.history.insert(tk.END, f"{it['ts']} | {it['verdict']} | {it['score']:.4f}")

    def _on_history_select(self, event):
        if not self.history.curselection():
            return
        line = self.history.get(self.history.curselection()[0])
        ts = line.split("|")[0].strip()
        row = get_result_by_ts(self.cfg, ts)
        if not row:
            return

        # load processed previews if exist
        imgs = find_processed_images(self.cfg, ts)
        if "norm" in imgs:
            self._set_thumbnail(self.norm_label, imgs["norm"])
        if "edges_closed" in imgs:
            self._set_thumbnail(self.edges_label, imgs["edges_closed"])

        # show text
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"TS: {row['ts']}\nImage: {row['image_path']}\n\n")
        self.result_text.insert(tk.END, f"ROI: x={row['roi_x']} y={row['roi_y']} w={row['roi_w']} h={row['roi_h']}\n\n")
        self.result_text.insert(tk.END, f"Brightness:  {row['brightness']:.1f}\n")
        self.result_text.insert(tk.END, f"Glare ratio: {row['glare_ratio']:.3f}\n")
        self.result_text.insert(tk.END, f"Sharpness:   {row['sharpness']:.1f}\n\n")
        self.result_text.insert(tk.END, f"Edge density: {row['edge_density']:.4f}\n")
        self.result_text.insert(tk.END, f"Continuity:   {row['continuity']:.1f}\n")
        self.result_text.insert(tk.END, f"Score:        {row['score']:.4f}\n")
        self.result_text.insert(tk.END, f"\nVERDICT: {row['verdict']}\n")
        if row.get("notes"):
            self.result_text.insert(tk.END, f"\nNotes: {row['notes']}\n")

    def _export_csv(self):
        p = export_csv(self.cfg)
        messagebox.showinfo("Export", f"Exported: {p}")

def run_app(cfg):
    app = TireGuardApp(cfg)
    app.mainloop()
