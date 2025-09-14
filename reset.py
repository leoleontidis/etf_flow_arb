#!/usr/bin/env python
"""
Reset utility for a fresh run:
- deletes everything under results/
- deletes all generated data files under data/, but keeps data/raw/ intact
"""

import os, shutil

def safe_rmdir(path: str, keep_raw: bool = False):
    if not os.path.exists(path):
        return
    if keep_raw:
        for item in os.listdir(path):
            full = os.path.join(path, item)
            if os.path.isdir(full) and item.lower() == "raw":
                continue  # keep raw
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                os.remove(full)
    else:
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

def main():
    print("[reset] clearing results/")
    safe_rmdir("results", keep_raw=False)

    print("[reset] clearing data/ but preserving data/raw/")
    safe_rmdir("data", keep_raw=True)

    print("[reset] done")

if __name__ == "__main__":
    main()
