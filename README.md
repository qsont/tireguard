# TireGuard

TireGuard is a comprehensive tire tread capture and inspection system featuring both a desktop UI and a FastAPI-powered web dashboard. It captures images, performs automated quality assessments, measures tread groove visibility, and provides tools for reviewing and exporting results.

---

## Features

- **Dual Interface**: Operate via a feature-rich desktop application or a responsive web dashboard.
- **Intelligent Capture**: Capture tire images with configurable Regions of Interest (ROI).
- **Automated Quality Control**: Performs checks on brightness, glare, and sharpness.
- **Tread Analysis**: Measures groove visibility, edge density, and continuity.
- **Decision Engine**: Generates verdicts (PASS/WARNING/REPLACE) based on calculated scores.
- **Session Management**: Records metadata such as vehicle ID, tire position, operator, and notes.
- **Data Persistence**: Stores results and processed images in a local SQLite database.
- **Export Capabilities**: Exports results to CSV for further analysis.
- **Calibration Support**: Includes tools for image scale calibration.

---

## Repository Structure

```
├── app.py                 # Entry point for the desktop application
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── calib/               # Calibration utilities (deprecated/labs)
│   ├── calibrate_laser_plane.py
│   └── calibrate_scale.py
├── data/                # Default data storage directory
│   ├── calibration.json # Calibration settings
│   ├── results_export.csv # CSV export destination
│   ├── roi.json         # Saved ROI coordinates
│   ├── captures/        # Raw captured images
│   └── processed/       # Preprocessed images for analysis
└── tireguard/           # Core application package
    ├── api.py           # FastAPI backend and web dashboard
    ├── auto_roi.py      # Automatic ROI suggestion algorithm
    ├── calibration.py   # Calibration logic (save/load/compute)
    ├── camera.py        # Camera interface and frame capture
    ├── config.py        # Application configuration
    ├── live_metrics.py  # Real-time metrics for UI feedback
    ├── measure.py       # Groove visibility and metric calculations
    ├── preprocess.py    # Image preprocessing pipeline (CLAHE, etc.)
    ├── quality.py       # Quality check algorithms
    ├── storage.py       # Data persistence (SQLite, file I/O)
    ├── ui.py            # UI framework abstraction layer
    ├── ui_qt.py         # PySide6 desktop UI implementation
    └── ui_tk.py         # Tkinter desktop UI implementation (legacy)
```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for a full list of dependencies.

---

## Setup

1. **Clone the repository** (or download and extract the source code).
2. **Create a virtual environment**:
    ```bash
    python -m venv .venv
    ```
3. **Activate the virtual environment**:
    - On Linux/macOS:
        ```bash
        source .venv/bin/activate
        ```
    - On Windows:
        ```cmd
        .venv\Scripts\activate
        ```
4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Running the Desktop Application

The desktop application provides the primary capture and inspection workflow.

1. Navigate to your project directory in the terminal.
2. Ensure your virtual environment is activated.
3. Run the entry point script:
    ```bash
    python app.py
    ```
   This will launch the PySide6 desktop UI by default (configurable in `tireguard.ui`).

### Running the Web Dashboard

The web dashboard offers a remote-accessible interface for reviewing scans and results.

1. Navigate to your project directory in the terminal.
2. Ensure your virtual environment is activated.
3. Start the FastAPI server:
    ```bash
    python -m tireguard.api
    ```
4. Open your web browser and navigate to:
    ```
    http://127.0.0.1:8000/
    ```
   The dashboard will display recent scans, allow filtering, and provide detailed views of individual inspections.

---

## Core Workflow (Technical Overview)

1. **Capture**: A frame is acquired from the camera using `tireguard.camera.open_camera`.
2. **Preprocessing**: The Region of Interest (ROI) is extracted and processed (e.g., CLAHE enhancement) via `tireguard.preprocess.preprocess_bgr`.
3. **Quality Assessment**: The processed image undergoes checks for brightness, glare, and sharpness using `tireguard.quality.run_quality_checks`.
4. **Tread Measurement**: Groove visibility and related metrics are calculated by `tireguard.measure.groove_visibility_score`.
5. **Storage**:
   - The original capture and processed images are saved using `tireguard.storage.save_capture` and `tireguard.storage.save_processed`.
   - A summary record containing metadata, metrics, and verdict is inserted into the SQLite database via `tireguard.storage.insert_result`.

---

## Data & Storage

By default, all persistent data is stored in the `./data` directory relative to the application root.

- **Raw Captures**: `data/captures/`
- **Processed Images**: `data/processed/`
- **ROI Coordinates**: `data/roi.json`
- **Calibration Data**: `data/calibration.json`
- **Database**: Configured within `tireguard.config.AppConfig` (typically `data/tireguard.db`).
- **CSV Export**: `data/results_export.csv` (generated by `tireguard.storage.export_csv`).

---

## API Endpoints (Web Dashboard)

The FastAPI application serves the dashboard and provides a RESTful API for data access. Key endpoints include:

- `GET /` - Serves the main dashboard HTML page.
- `GET /api/health` - Simple service health check.
- `GET /api/scans?limit=50` - Fetches a paginated list of recent scans.
- `GET /api/scans/{ts}` - Retrieves detailed information for a specific scan timestamp.
- `GET /api/scans/{ts}/images` - Lists available processed image types for a scan.
- `GET /api/images/{ts}/{kind}` - Serves a specific processed image (e.g., 'gray', 'edges').
- `GET /api/export/csv` - Triggers and serves a CSV export of all results.

---

## Calibration

The system supports image scale calibration to provide measurements in physical units (e.g., mm).

- Calibration settings are managed by `tireguard.calibration.load_calibration` and `tireguard.calibration.save_calibration`.
- A two-point calibration method (`tireguard.calibration.compute_scale_from_two_points`) allows setting a scale factor by clicking two points of known distance in the captured image.

---

## ROI (Region of Interest)

The ROI defines the area of the tire image analyzed for tread quality.

- ROIs can be manually set using the desktop UI or suggested automatically (`tireguard.auto_roi.suggest_roi`).
- Images are cropped to the ROI before processing (`tireguard.preprocess.crop_roi`).
- The active ROI coordinates are typically saved to `data/roi.json`.

---

## Troubleshooting

- **Empty fields in the dashboard table**: This often indicates that the `list_results` function in `tireguard.storage` is not returning all the columns expected by the web frontend's table structure. Ensure the query in `list_results` selects all required fields (e.g., operator, brightness, sharpness, etc.).
- **Images not showing in the dashboard**: Verify that the `processed` images are being saved correctly to the `data/processed` directory by the `save_processed` function. Check the paths generated by `find_processed_images`.
- **Camera not working**: Check the camera index and resolution settings in `tireguard.config.AppConfig` and ensure the specified camera device is available and not in use by another application.