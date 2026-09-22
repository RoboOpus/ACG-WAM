"""Export vector Figs. 1 and 2 and remux the project video without re-encoding.

Requires pymupdf and imageio-ffmpeg. Original source files are never modified.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import imageio_ffmpeg
import pymupdf
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--figure', type=Path, required=True)
parser.add_argument('--video', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
images = repo/'assets/images'
images.mkdir(parents=True, exist_ok=True)
for figure_source,figure_name,trim in [(args.figure,'architecture',(56,67,56,59)),(args.figure.with_name('teaser_v10.pdf'),'overview',(39,212,290,64))]:
    with pymupdf.open(figure_source) as doc:
        page = doc[0]
        # Match each figure's manuscript crop without rasterizing.
        left,bottom,right,top = trim
        page.set_cropbox(pymupdf.Rect(left,top,page.rect.width-right,page.rect.height-bottom))
        svg = page.get_svg_image(text_as_path=True)
        # Bake each raster soft mask into PNG alpha. Keep all text/linework vector;
        # avoid renderer-dependent SVG mask handling around the robot and icons.
        ns = 'http://www.w3.org/2000/svg'
        href = '{http://www.w3.org/1999/xlink}href'
        ET.register_namespace('', ns)
        ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
        tree = ET.fromstring(svg)
        ids = {node.get('id'):node for node in tree.iter() if node.get('id')}
        for group in tree.iter('{'+ns+'}g'):
            if 'mask' not in group.attrib:
                continue
            mask = ids[group.get('mask')[5:-1]]
            def image_node(container):
                leaves = [node for node in container.iter() if node.tag in ('{'+ns+'}image','{'+ns+'}use')]
                if len(leaves) != 1:
                    raise ValueError('Unexpected SVG mask structure')
                node = leaves[0]
                return ids[node.get(href)[1:]] if node.tag.endswith('use') else node
            mask_image = image_node(mask)
            target_image = image_node(group)
            mask_pixels = Image.open(io.BytesIO(base64.b64decode(mask_image.get(href).split(',',1)[1]))).convert('L')
            target_pixels = Image.open(io.BytesIO(base64.b64decode(target_image.get(href).split(',',1)[1]))).convert('RGBA')
            if mask_pixels.size != target_pixels.size:
                raise ValueError('SVG image/mask size mismatch')
            target_pixels.putalpha(mask_pixels)
            buffer = io.BytesIO()
            target_pixels.save(buffer, format='PNG')
            target_image.set(href, 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
            del group.attrib['mask']
        svg = ET.tostring(tree,encoding='unicode')
        (images/f'{figure_name}.svg').write_text(svg,encoding='utf8')
    shutil.copyfile(figure_source,images/f'{figure_name}.pdf')

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
video = repo/'assets/videos/project-overview.mp4'
poster = video.with_suffix('.jpg')
commands = [
    ['-i',str(args.video),'-map','0:v:0','-map','0:a?','-c','copy','-map_metadata','-1','-movflags','+faststart','-y',str(video)],
    ['-ss','3','-i',str(video),'-frames:v','1','-q:v','2','-y',str(poster)],
]
for command in commands:
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error',*command],check=True)
metadata = {
    'overview_figure': {'source':'teaser_v10.pdf','svg':'assets/images/overview.svg',
                        'pdf':'assets/images/overview.pdf','source_sha256':hashlib.sha256(args.figure.with_name('teaser_v10.pdf').read_bytes()).hexdigest(),
                        'export':'Vector paths and embedded original images; fonts outlined; manuscript crop.'},
    'figure': {'source':args.figure.name,'svg':'assets/images/architecture.svg',
               'pdf':'assets/images/architecture.pdf','source_sha256':hashlib.sha256(args.figure.read_bytes()).hexdigest(),
               'export':'Vector paths and embedded original images; fonts outlined; manuscript crop.'},
    'overview_video': {'source':args.video.name,'src':video.relative_to(repo).as_posix(),
                      'poster':poster.relative_to(repo).as_posix(),
                      'source_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest(),
                      'bytes':video.stat().st_size,'processing':'Stream copy with original H.264 video and AAC audio; MP4 faststart.'},
    'real_demo_selection': {
        'Bimanual Fruit Placement': ['1/IMG_6668.mp4','1/IMG_6670.mp4'],
        'Block Stacking': {'robot_left_arm':'2/IMG_6661.mp4','robot_right_arm':'2/IMG_6665.mp4'},
        'Toy Placement into a Cup': {'robot_left_arm':'3/IMG_6651.mp4','robot_right_arm':'3/IMG_6649.mp4'},
        'arm_convention':'Left/right is from the robot perspective, opposite the frontal camera view.'
    }
}
(repo/'assets/presentation-manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
print('Vector figures:', {name:(images/f'{name}.svg').stat().st_size for name in ('overview','architecture')}, 'bytes')
print('Project video:',video.stat().st_size,'bytes; original audio retained')
