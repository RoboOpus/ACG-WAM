"""Prepare web assets without modifying originals.

pip install imageio-ffmpeg pymupdf pillow
python tools/prepare_media.py --source-root C:/path/to/BaiduSyncdisk
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import imageio_ffmpeg
import pymupdf
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
root = args.source_root.resolve()
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
real_root = root / 'tmp/figure_motus/video_infer/success'
sim_root = root / 'tmp/figure_motus/robotwin_LDJEPA_50task_selected6_videos'
paper_root = root / 'latex_workspace/icra2027/paper/icra2026'

def run(command):
    result = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', *command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)

def prepare(source):
    is_real = source.suffix.lower() == '.mov'
    relative = source.relative_to(real_root if is_real else sim_root)
    target = repo / 'assets/videos' / ('real' if is_real else 'simulation') / relative.with_suffix('.mp4')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.tmp.mp4')
        if is_real:
            # iPhone HLG / BT.2020 -> SDR / BT.709, full duration at original speed.
            filters = 'fps=30,scale=1920:-2,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p'
            options = ['-vf', filters, '-c:v', 'libx264', '-preset', 'medium', '-crf', '24', '-threads', '4', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709']
        else:
            # Already H.264 at 320x240, 10 fps. Remux without quality loss.
            options = ['-c:v', 'copy']
        run(['-i', str(source), '-map', '0:v:0', '-an', '-map_metadata', '-1', *options, '-movflags', '+faststart', '-y', str(temporary)])
        temporary.replace(target)
    poster = target.with_suffix('.jpg')
    if not poster.exists():
        run(['-ss', '1', '-i', str(target), '-frames:v', '1', '-vf', 'scale=640:-2' if is_real else 'scale=320:-2', '-q:v', '3', '-y', str(poster)])
    info = subprocess.run([ffmpeg, '-hide_banner', '-i', str(target)], capture_output=True, text=True).stderr
    stream = next(line.strip() for line in info.splitlines() if 'Video:' in line)
    if 'h264' not in stream or 'yuv420p' not in stream:
        raise RuntimeError(f'Unexpected output format: {stream}')
    duration = re.search(r'Duration: (\d+):(\d+):([\d.]+)', info)
    seconds = sum(float(v)*m for v,m in zip(duration.groups(), [3600,60,1]))
    entry = dict(kind='real' if is_real else 'simulation', source=relative.as_posix(), src=target.relative_to(repo).as_posix(), poster=poster.relative_to(repo).as_posix(), original_bytes=source.stat().st_size, bytes=target.stat().st_size, duration_seconds=seconds, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), codec=stream)
    print(f'{relative}: {entry["original_bytes"]/1024**2:.2f} -> {entry["bytes"]/1024**2:.2f} MiB', flush=True)
    return entry

sources = sorted(real_root.rglob('*.MOV')) + sorted(sim_root.rglob('*.mp4'))
if len(sources) != 34:
    raise RuntimeError(f'Expected 10 real and 24 simulation videos; found {len(sources)}')
with ThreadPoolExecutor(max_workers=2) as executor:
    manifest = list(executor.map(prepare, sources))
(repo/'assets/media-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')

figures = [('teaser_v10.pdf', 'overview', (39,212,290,64)), ('figure_architecture_v4.pdf', 'architecture', (56,67,56,59)), ('figure_realworld_v7.pdf', 'real-world', (7,124,35,5))]
out = repo/'assets/images'
out.mkdir(parents=True, exist_ok=True)
for filename, name, (left,bottom,right,top) in figures:
    with pymupdf.open(paper_root/'figures/figures_all'/filename) as doc:
        page = doc[0]
        clip = pymupdf.Rect(left, top, page.rect.width-right, page.rect.height-bottom)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2400/clip.width,2400/clip.width), clip=clip, alpha=False)
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        im.save(out/f'{name}.webp', quality=92, method=6)
        print(f'Figure: {name} {im.size}', flush=True)
(repo/'assets/paper').mkdir(exist_ok=True)
shutil.copy2(paper_root/'manuscript/main.pdf', repo/'assets/paper/acg-wam.pdf')
print('All media ready.', flush=True)
