"""
download_dataset.py
--------------------
Optional helper: download images from a list of direct URLs into real/ and
screen/ folders, so train.py has data to work with.

Usage:
    1. Put direct image URLs (one per line) into two text files:
         real_urls.txt
         screen_urls.txt
    2. Run:
         python download_dataset.py --real-urls real_urls.txt --screen-urls screen_urls.txt

    Or, if you already have local image files, SKIP this script entirely --
    just put them straight into real/ and screen/ folders yourself
    (this is the normal, recommended path).

Notes:
- Google Drive "share" links (drive.google.com/file/d/.../view) are NOT direct
  image links and will not download correctly with this script. Convert them
  first, e.g. https://drive.google.com/uc?export=download&id=FILE_ID
  or simply download the files from Drive to your computer and drop them in
  the real/ and screen/ folders manually -- that's simpler and more reliable.
"""

import argparse
import os
import sys
import urllib.request

import os
import re


def safe_filename(url: str, idx: int) -> str:
    ext = ".jpg"
    m = re.search(r"\.(jpg|jpeg|png|webp|bmp)(\?|$)", url, re.IGNORECASE)
    if m:
        ext = "." + m.group(1).lower()
    return f"img_{idx:03d}{ext}"


def download_list(url_file: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    ok, fail = 0, 0
    for i, url in enumerate(urls, 1):
        fname = safe_filename(url, i)
        out_path = os.path.join(out_dir, fname)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
            ok += 1
            print(f"  [{i}/{len(urls)}] OK   -> {fname}")
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(urls)}] FAIL -> {url}  ({e})", file=sys.stderr)

    print(f"\n  {out_dir}: {ok} downloaded, {fail} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-urls",   default="real_urls.txt")
    ap.add_argument("--screen-urls", default="screen_urls.txt")
    ap.add_argument("--real-dir",    default="real")
    ap.add_argument("--screen-dir",  default="screen")
    args = ap.parse_args()

    if os.path.exists(args.real_urls):
        print(f"Downloading REAL images from {args.real_urls} ...")
        download_list(args.real_urls, args.real_dir)
    else:
        print(f"SKIP: {args.real_urls} not found")

    if os.path.exists(args.screen_urls):
        print(f"\nDownloading SCREEN images from {args.screen_urls} ...")
        download_list(args.screen_urls, args.screen_dir)
    else:
        print(f"SKIP: {args.screen_urls} not found")


if __name__ == "__main__":
    main()
