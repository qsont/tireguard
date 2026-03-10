"""
Single UI entrypoint.
Switch UI_IMPL to choose which UI runs without changing app.py.
"""
UI_IMPL = "qt_wizard"   # options: "qt_wizard" (default), "qt_basic"

def run_app(cfg, simple_ui=False, compact_ui=None, fullscreen=False, rpi_ui=False):
    if simple_ui:
        from .ui_qt_touch import run_app as _run
        return _run(cfg)
    if UI_IMPL == "qt_basic":
        from .ui_qt_basic import run_app as _run
        return _run(cfg)
    else:
        from .ui_qt import run_app as _run
        try:
            return _run(cfg, compact_ui=compact_ui, fullscreen=fullscreen, rpi_ui=rpi_ui)
        except TypeError:
            # Backward-compatible fallback for older UI implementations.
            return _run(cfg)
