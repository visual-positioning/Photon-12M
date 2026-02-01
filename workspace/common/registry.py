import os
import pkgutil
import importlib

def scan_directory(folder_name):
    """Scans for valid plugin modules, ignoring __init__ and private files."""
    valid_modules = []
    path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(path): return []
    for _, name, _ in pkgutil.iter_modules([path]):
        if not name.startswith("_") and name != "custom":
            valid_modules.append(name)
    return valid_modules

def dynamic_import(module_path, class_name):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # In research, silence is golden, but we log import errors to stderr
        import sys
        print(f"[Import Error] {class_name} in {module_path}: {e}", file=sys.stderr)
        return None