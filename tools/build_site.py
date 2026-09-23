"""Generate the ACG-WAM project page. Run with Python 3."""
from pathlib import Path
from html import escape
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'assets/media-manifest.json').read_text(encoding='utf8'))
presentation = json.loads((ROOT/'assets/presentation-manifest.json').read_text(encoding='utf8'))

def video(src, poster, label, overview=False):
    playback = 'autoplay muted loop preload="metadata"'
    return f'<video controls playsinline {playback} poster="{poster}" aria-label="{escape(label)}"><source src="{src}" type="video/mp4">Your browser does not support embedded video. <a href="{src}">Download the video</a>.</video>'

real_tasks = [
    ('Bimanual Fruit Placement', [('1/IMG_6668.MOV','Demonstration 1'),('1/IMG_6670.MOV','Demonstration 2')]),
    ('Block Stacking', [('2/IMG_6661.MOV','Left arm'),('2/IMG_6665.MOV','Right arm')]),
    ('Toy Placement into a Cup', [('3/IMG_6651.MOV','Left arm'),('3/IMG_6649.MOV','Right arm')]),
]
real_rows=[]
for task_index,(title,choices) in enumerate(real_tasks):
    cards=[]
    for source,label in choices:
        entry=next(v for v in manifest if v['kind']=='real' and v['source']==source)
        cards.append(f'<article class="demo" data-real-demo><h4>{label}</h4>{video(entry["src"],entry["poster"],title+" — "+label)}</article>')
    real_rows.append(f'<section class="task-row" aria-labelledby="real-task-{task_index}"><h3 id="real-task-{task_index}">{title}</h3><div class="real-grid">{"".join(cards)}</div></section>')

sim_tasks=[('HangingMug','Hanging Mug'),('HandoverMic','Handover Mic'),('MoveCanPot','Move Can Pot'),('PlaceMousePad','Place Mouse Pad'),('ScanObject','Scan Object'),('OpenMicrowave','Open Microwave')]
sim_cards=[]
for slug,title in sim_tasks:
    base=f'assets/videos/simulation/{slug}'
    sim_cards.append(f'<article class="demo" data-sim-card data-base="{base}" data-title="{title}"><h3>{title}</h3>{video(base+"/demo_clean/episode0.mp4",base+"/demo_clean/episode0.jpg",title+" — clean, demonstration 1")}</article>')

bib='''@unpublished{liu2026acgwam,
  title  = {ACG-WAM: World Action Modeling via Action Conditioned Geometric Latent Prediction},
  author = {Liu, Jiangtao and Xiang, Zishang and He, Yage and Cui, Lingguo and Zhang, Baihai and Chai, Runqi and Chai, Senchun},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://RoboOpus.github.io/ACG-WAM/}
}'''
overview=presentation['overview_video']
css_version=hashlib.sha256((ROOT/'assets/site.css').read_bytes()).hexdigest()[:12]
js_version=hashlib.sha256((ROOT/'assets/site.js').read_bytes()).hexdigest()[:12]
html=f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ACG-WAM: World Action Modeling via Action Conditioned Geometric Latent Prediction</title>
<meta name="description" content="ACG-WAM project video, method, real robot demonstrations and RoboTwin simulation.">
<meta name="theme-color" content="#ffffff">
<meta property="og:title" content="ACG-WAM: World Action Modeling">
<meta property="og:description" content="World Action Modeling via Action Conditioned Geometric Latent Prediction. Project video and robot demonstrations.">
<meta property="og:type" content="website"><meta property="og:url" content="https://RoboOpus.github.io/ACG-WAM/">
<meta property="og:image" content="https://RoboOpus.github.io/ACG-WAM/assets/videos/project-overview.jpg">
<link rel="canonical" href="https://RoboOpus.github.io/ACG-WAM/"><link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/site.css?v={css_version}"><script src="assets/site.js?v={js_version}" defer></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar"><div class="shell"><a class="brand" href="#top">ACG-WAM</a><nav aria-label="Main navigation"><a href="#overview">Project video</a><a href="#method">Method</a><a href="#real-world">Real robot</a><a href="#simulation">Simulation</a></nav></div></header>
<main id="main">
<section class="hero shell" id="top" aria-labelledby="paper-title">
<h1 id="paper-title"><span class="title-line"><span class="project-name">ACG-WAM:</span> World Action Modeling</span><span class="title-line">via Action Conditioned Geometric Latent Prediction</span></h1>
<p class="authors"><span>Jiangtao Liu<sup>*</sup></span><span>Zishang Xiang<sup>*</sup></span><span>Yage He</span><span>Lingguo Cui</span><span>Baihai Zhang</span><span>Runqi Chai</span><span>Senchun Chai<sup>†</sup></span></p>
<p class="author-note"><sup>*</sup> Equal contribution. &nbsp; <sup>†</sup> Corresponding author.</p>
<div class="links"><a class="button primary" href="assets/paper/acg-wam.pdf">Paper</a><a class="button" href="https://github.com/RoboOpus/ACG-WAM">Code</a><a class="button" href="https://huggingface.co/RoboOpus/ACG-WAM">Model</a></div>
<div class="overview-video" id="overview">{video(overview['src'],overview['poster'],'ACG-WAM project overview',overview=True)}</div>
</section>
<section class="section shell" id="method" aria-labelledby="method-title">
<h2 id="method-title">Method Overview</h2>
<figure class="method-row"><a class="method-image" href="assets/images/overview.svg" aria-label="Open full size Fig. 1"><img src="assets/images/overview.svg" width="2401" height="1196" loading="lazy" decoding="async" alt="Figure 1. ACG-WAM combines geometric prediction conditioned on actions with distillation across camera views, evaluated in simulation and on a real robot."></a><figcaption class="method-copy"><h3>Geometric Prediction Conditioned on Actions</h3><p>ACG-WAM learns to predict future geometric representations from current observations and intervening actions. Our ACG-JEPA objective combines prediction over multiple horizons with geometric distillation from head and wrist cameras to train the policy’s shared visual embedding.</p><p>We evaluate ACG-WAM on 50 RoboTwin 2.0 tasks and three real robot tasks. Ablations on six simulation tasks examine the joint geometric targets, action conditioning, and supervision at multiple horizons.</p></figcaption></figure>
<figure class="method-row method-row-reverse"><a class="method-image" href="assets/images/architecture.svg" aria-label="Open full size Fig. 2"><img src="assets/images/architecture.svg" width="848" height="414" loading="lazy" decoding="async" alt="Figure 2. ACG-JEPA supervises the MoT backbone’s shared visual embedding by predicting geometric targets from head and wrist cameras, conditioned on actions over multiple horizons."></a><figcaption class="method-copy"><h3>ACG-WAM Architecture</h3><p><strong>Prediction conditioned on actions.</strong> ACG-JEPA predicts geometric targets from features of the current frame, intervening actions, and a temporal horizon. A frozen VGGT teacher jointly encodes pairs of current and future images and supplies targets from the future slot at multiple horizons.</p><p><strong>Geometric distillation across views.</strong> Targets from head and wrist cameras supervise the MoT backbone’s shared visual embedding before temporal mixing, so the predictor’s visual input contains only the current observation. The geometric loss updates this embedding alongside the video and action objectives; the teacher and auxiliary predictor are removed at inference.</p></figcaption></figure>
</section>
<section class="section shell" id="real-world" aria-labelledby="real-title">
<h2 id="real-title">Real Robot Demonstrations</h2>
<p>ACG-WAM performs bimanual fruit placement, block stacking, and toy placement into a cup on TRON2 with WUJI hands. Block stacking and toy placement are shown with both the robot’s left and right arms.</p>
{''.join(real_rows)}
</section>
<section class="section shell" id="simulation" aria-labelledby="sim-title">
<h2 id="sim-title">RoboTwin Simulation</h2>
<p>Demonstrations across six manipulation tasks in clean and randomized scenes.</p>
<div class="toolbar"><div class="segmented" role="group" aria-label="Simulation scene setting"><button type="button" data-scene="clean" aria-pressed="true">Clean</button><button type="button" data-scene="randomized" aria-pressed="false">Randomized</button></div><div class="segmented" role="group" aria-label="Simulation demonstration"><button type="button" data-episode="0" aria-pressed="true">Demo 1</button><button type="button" data-episode="1" aria-pressed="false">Demo 2</button></div></div>
<p class="sr-only" id="scene-status" role="status">Clean scenes, demonstration 1</p>
<div class="simulation-grid">{''.join(sim_cards)}</div>
<noscript><p>Enable JavaScript to switch scenes and demonstrations, or <a href="https://github.com/RoboOpus/ACG-WAM/tree/website/assets/videos/simulation">browse the simulation videos</a>.</p></noscript>
</section>
<section class="section shell" id="citation" aria-labelledby="citation-title">
<div class="citation-heading"><h2 id="citation-title">Citation</h2><button class="button" id="copy-citation" type="button">Copy BibTeX</button></div>
<pre><code id="bibtex">{escape(bib)}</code></pre><p id="copy-status" role="status"></p>
</section>
</main>
<footer class="shell"><a href="https://github.com/RoboOpus/ACG-WAM">ACG-WAM</a><a href="#top">Back to top</a></footer>
</body></html>'''
(ROOT/'index.html').write_text(html,encoding='utf8')
print('Built demo-focused page: project video, vector Figs. 1 and 2, six real-robot demos and 24 simulation choices.')
