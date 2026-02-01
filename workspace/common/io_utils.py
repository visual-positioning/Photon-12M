"""
Cross-platform IO utilities for the evaluation framework.
Works on both Windows and Unix systems.
"""
import json
import os
import sys

# Cross-platform file locking
if sys.platform == 'win32':
    import msvcrt
    
    def lock_file(f):
        """Acquire exclusive lock on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    
    def unlock_file(f):
        """Release lock on Windows."""
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # Lock may already be released
else:
    import fcntl
    
    def lock_file(f):
        """Acquire exclusive lock on Unix."""
        fcntl.flock(f, fcntl.LOCK_EX)
    
    def unlock_file(f):
        """Release lock on Unix."""
        fcntl.flock(f, fcntl.LOCK_UN)


def append_to_json(filepath, entry):
    """
    Thread-safe append to JSON file.
    Works on both Windows and Unix systems.
    
    Args:
        filepath: Path to JSON file (will be created if doesn't exist)
        entry: Dictionary entry to append to the JSON array
    """
    # Create file with empty array if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump([], f)

    # Read, Append, Write strategy with file locking
    with open(filepath, 'r+') as f:
        try:
            lock_file(f)
            
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            
            if not isinstance(data, list):
                data = [data]
            
            data.append(entry)
            
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        finally:
            unlock_file(f)