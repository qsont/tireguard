"""
Single UI entrypoint.
Switch UI_IMPL to choose which UI runs without changing app.py.
"""
UI_IMPL = "qt_wizard"   # options: "qt_wizard" (default), "qt_basic"

def run_app(cfg):
    if UI_IMPL == "qt_basic":
        from .ui_qt_basic import run_app as _run
        return _run(cfg)
    else:
        from .ui_qt import run_app as _run
        return _run(cfg)
