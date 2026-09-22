"""Generate the static project page. Run from any directory with Python 3."""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'assets/media-manifest.json').read_text(encoding='utf8'))
real_tasks = [
    ('1', 'Bimanual Fruit Placement', 'Coordinated grasping and placement of two fruits.', '85% SR · 92.5% PCS'),
    ('2', 'Block Stacking', 'Place and release the upper block into a stable stack.', '75% SR · 85.0% PCS'),
    ('3', 'Toy Placement into a Cup', 'Grasp a toy and place it inside a confined receptacle.', '95% SR · 97.5% PCS'),
]
sim_tasks = [('HangingMug','Hanging Mug',70,76),('HandoverMic','Handover Mic',100,98),('MoveCanPot','Move Can Pot',98,90),('PlaceMousePad','Place Mouse Pad',94,81),('ScanObject','Scan Object',92,87),('OpenMicrowave','Open Microwave',98,99)]

def video(src, poster, label, hero=False):
    return f'<video controls muted loop playsinline preload="{"metadata" if hero else "none"}" {"data-hero" if hero else ""} poster="{poster}" aria-label="{escape(label)}"><source src="{src}" type="video/mp4">Your browser does not support embedded video. <a href="{src}">Download the video</a>.</video>'

real_cards=[]
for number,title,description,score in real_tasks:
    entries=[v for v in manifest if v['kind']=='real' and v['source'].startswith(number+'/')]
    first=entries[0]
    options=''.join(f'<option value="{e["src"]}" data-poster="{e["poster"]}">Sample {i+1} · {e["duration_seconds"]:.1f}s</option>' for i,e in enumerate(entries))
    real_cards.append(f'''<article class="video-card" data-real-card>
      {video(first['src'],first['poster'],title)}
      <div class="card-body"><div class="card-top"><span class="task-number">0{number} / REAL ROBOT</span><span class="badge">Original speed</span></div>
      <h3>{title}</h3><p>{description}</p><p>{score}</p>
      <label>Demonstration<select aria-label="{title} sample" data-real-select>{options}</select></label></div></article>''')

sim_cards=[]
for i,(slug,title,clean,randomized) in enumerate(sim_tasks):
    base=f'assets/videos/simulation/{slug}'
    sim_cards.append(f'''<article class="video-card" data-sim-card data-base="{base}" data-title="{title}" data-clean="{clean}" data-randomized="{randomized}">
      {video(base+'/demo_clean/episode0.mp4',base+'/demo_clean/episode0.jpg',title+' — clean, episode 0')}
      <div class="card-body"><div class="card-top"><span class="task-number">0{i+1} / SIMULATION</span><span class="badge" data-scene-label>Clean</span></div>
      <h3>{title}</h3><p data-score>Clean success rate: {clean}%</p><label>Demonstration<select data-episode aria-label="{title} episode"><option value="0">Episode 0</option><option value="1">Episode 1</option></select></label></div></article>''')

sim_results=[('GO-1','37.80','36.24','37.02'),('π₀.₅','42.98','43.84','43.41'),('X-VLA','72.80','72.84','72.82'),('Motus','88.66','87.02','87.84'),('LingBot-VA','91.99','91.11','91.55'),('MECo-WAM','93.26','91.98','92.62'),('WAM4D','<strong>93.82</strong>','89.86','91.84'),('ACG-WAM','93.46','<strong>92.68</strong>','<strong>93.07</strong>')]
real_results=[('π₀.₅','65.00','75.00'),('LingBot-VA','78.33','84.17'),('Motus','75.00','82.50'),('ACG-WAM','<strong>85.00</strong>','<strong>91.67</strong>')]
def rows(data):
    output = []
    for row in data:
        css = ' class="ours"' if row[0] == 'ACG-WAM' else ''
        cells = f'<th scope="row">{row[0]}</th>'
        cells += ''.join(f'<td>{value}</td>' for value in row[1:])
        output.append(f'<tr{css}>{cells}</tr>')
    return ''.join(output)

bib='''@unpublished{liu2026acgwam,
  title  = {ACG-WAM: World-Action Modeling via
            Action-Conditioned Geometric Latent Prediction},
  author = {Liu, Jiangtao and Xiang, Zishang and He, Yage and
            Cui, Lingguo and Zhang, Baihai and Chai, Runqi and Chai, Senchun},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://RoboOpus.github.io/ACG-WAM/}
}'''
hero=next(v for v in manifest if v['source']=='1/IMG_6668.MOV')
html=f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ACG-WAM · Action-Conditioned Geometric Latent Prediction</title>
<meta name="description" content="ACG-WAM: World-Action Modeling via Action-Conditioned Geometric Latent Prediction. Method, manuscript, real-robot demonstrations, and RoboTwin 2.0 results.">
<meta name="theme-color" content="#0c1418"><meta property="og:title" content="ACG-WAM: World-Action Modeling">
<meta property="og:description" content="Learning the geometric consequences of actions. Explore ACG-JEPA, real-robot demonstrations, and results across 50 RoboTwin 2.0 tasks.">
<meta property="og:type" content="website"><meta property="og:url" content="https://RoboOpus.github.io/ACG-WAM/">
<meta property="og:image" content="https://RoboOpus.github.io/ACG-WAM/assets/images/overview.webp">
<link rel="canonical" href="https://RoboOpus.github.io/ACG-WAM/"><link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/site.css"><script src="assets/site.js" defer></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar"><div class="shell"><a class="brand" href="#top">ACG<span>—</span>WAM</a><nav aria-label="Main navigation"><a href="#method">Method</a><a href="#results">Results</a><a href="#real-world">Real robot</a><a href="#simulation">Simulation</a><a href="#citation">Citation</a></nav></div></header>
<main id="main">
<section class="hero shell" id="top" aria-labelledby="paper-title">
<div class="eyebrow">World-action models · Geometric representation learning</div>
<div class="hero-layout"><div><h1 id="paper-title"><span class="acronym">ACG-WAM</span>World-Action Modeling<span class="subtitle">via Action-Conditioned<br>Geometric Latent Prediction</span></h1>
<p class="lede">Learning the geometric consequences of actions.<br>Geometry guides training; the policy acts on its own.</p></div>
<div class="hero-media">{video(hero['src'],hero['poster'],'Bimanual fruit placement with ACG-WAM',True)}<div class="media-caption"><span>TRON2 × WUJI hands</span><span>Real robot / 1× speed</span></div></div></div>
<p class="authors"><span>Jiangtao Liu<sup>*</sup></span><span>Zishang Xiang<sup>*</sup></span><span>Yage He</span><span>Lingguo Cui</span><span>Baihai Zhang</span><span>Runqi Chai</span><span>Senchun Chai<sup>†</sup></span></p>
<p class="author-note"><sup>*</sup> Equal contribution. &nbsp; <sup>†</sup> Corresponding author.</p>
<div class="links"><a class="button primary" href="assets/paper/acg-wam.pdf">Paper PDF <span aria-hidden="true">↗</span></a><a class="button" href="https://github.com/RoboOpus/ACG-WAM">Code <span aria-hidden="true">↗</span></a><a class="button" href="https://huggingface.co/RoboOpus/ACG-WAM">Model <small>planned release</small><span aria-hidden="true">↗</span></a><a class="button" href="#real-world">Watch demonstrations <span aria-hidden="true">↓</span></a></div>
<div class="stats" aria-label="Key results"><div class="stat"><strong>93.07<span>%</span></strong><p>RoboTwin 2.0 · mean success</p></div><div class="stat"><strong>92.68<span>%</span></strong><p>RoboTwin 2.0 · randomized</p></div><div class="stat"><strong>85.00<span>%</span></strong><p>Real robot · mean success</p></div><div class="stat"><strong>+10.00<span> pp</span></strong><p>Real-robot success over Motus</p></div></div>
</section>
<section class="section shell" id="method" aria-labelledby="method-title"><div class="section-heading"><div><div class="eyebrow">01 / The idea</div><h2 id="method-title">Predict geometry. Learn to act.</h2></div><p>Action-conditioned geometric supervision, applied before temporal mixing. No auxiliary geometry modules at inference.</p></div>
<p class="intro">World-action models jointly predict video and robot actions, but their standard objectives provide no explicit target for the <strong>geometric consequences of an action sequence</strong>. ACG-WAM introduces <strong>ACG-JEPA</strong>: predict future geometric features from the current observation, the intervening actions, and a prediction horizon.</p>
<figure class="figure"><a href="assets/images/overview.webp" aria-label="Open full-size ACG-WAM overview"><img src="assets/images/overview.webp" width="2401" height="1196" loading="lazy" decoding="async" alt="ACG-WAM overview showing auxiliary geometric supervision during training, the policy-only inference path, and evaluation results."></a><figcaption>A frozen VGGT teacher provides geometric targets. Auxiliary gradients update the shared visual embedding during training; deployment uses the trained Motus backbone. Click the figure to enlarge.</figcaption></figure>
<div class="steps"><article class="step"><small>01 / CONDITION</small><h3>Actions connect observations</h3><p>The predictor receives current visual features, the demonstrated action interval, and a temporal horizon. Multiple horizons capture short interactions and longer transitions.</p></article><article class="step"><small>02 / SUPERVISE</small><h3>Geometry from multiple views</h3><p>A frozen VGGT teacher jointly encodes current and future images within each camera. The future slot supplies targets from the head and wrist views.</p></article><article class="step"><small>03 / DEPLOY</small><h3>Keep the original policy path</h3><p>Supervision reaches the shared visual embedding before temporal mixing. The teacher and auxiliary predictor are removed for inference.</p></article></div>
<details class="architecture"><summary>Explore the training architecture</summary><figure class="figure"><a href="assets/images/architecture.webp"><img src="assets/images/architecture.webp" width="2401" height="1173" loading="lazy" alt="Detailed ACG-JEPA architecture: current visual features, demonstrated actions and horizon enter the predictor; future images enter only the frozen teacher."></a><figcaption>Future images construct teacher targets without entering the student’s current visual input. The auxiliary loss trains the shared patch embedding alongside the base video and action objectives.</figcaption></figure></details></section>
<section class="section shell" id="results" aria-labelledby="results-title"><div class="section-heading"><div><div class="eyebrow">02 / Evaluation</div><h2 id="results-title">From simulation to the real world.</h2></div><p>50 RoboTwin 2.0 tasks and three physical manipulation tasks. All results below are reported in the manuscript.</p></div>
<div class="table-grid"><div><div class="table-wrap"><table><caption>RoboTwin 2.0 · success rate (%)</caption><thead><tr><th scope="col">Method</th><th scope="col">Clean</th><th scope="col">Randomized</th><th scope="col">Mean</th></tr></thead><tbody>{rows(sim_results)}</tbody></table></div><p class="note">Macro averages across 50 tasks. ACG-WAM uses 100 evaluation episodes per task per setting. Best column values are bold.</p></div>
<div class="result-aside"><div class="table-wrap"><table><caption>Real robot · task averages (%)</caption><thead><tr><th scope="col">Method</th><th scope="col">Success</th><th scope="col">PCS</th></tr></thead><tbody>{rows(real_results)}</tbody></table></div><div class="callout"><strong>91.67% partial completion</strong>Alongside 85.00% mean success, ACG-WAM improves partial completion by 9.17 percentage points over Motus.</div></div></div>
<p class="note">Real robot: 20 evaluation trials per task, 60 per policy. PCS is the normalized partial completion score; averages weight the three tasks equally. See the paper for baseline sources and the full protocol.</p></section>
<section class="section shell" id="real-world" aria-labelledby="real-title"><div class="section-heading"><div><div class="eyebrow">03 / Physical manipulation</div><h2 id="real-title">Real robot. Complete executions.</h2></div><p>TRON2 with WUJI hands. Ten successful demonstrations across three tasks, shown at original speed. Select a sample to explore each execution.</p></div>
<div class="video-grid">{''.join(real_cards)}</div><p class="note">These are selected successful demonstrations. The success rates above come from the complete evaluation protocol, including unsuccessful trials. Videos are muted; use the player controls to play or enter fullscreen.</p>
<details class="architecture"><summary>View the real-world evaluation figure</summary><figure class="figure"><a href="assets/images/real-world.webp"><img src="assets/images/real-world.webp" width="2401" height="1206" loading="lazy" alt="Real-world evaluation overview and keyframe sequences for fruit placement, block stacking, and toy placement into a cup."></a><figcaption>The real-world evaluation figure from the manuscript.</figcaption></figure></details></section>
<section class="section shell" id="simulation" aria-labelledby="sim-title"><div class="section-heading"><div><div class="eyebrow">04 / RoboTwin 2.0</div><h2 id="sim-title">Six tasks. Two scene settings.</h2></div><p>Explore 24 recorded rollouts: two episodes per task in clean scenes and randomized scenes.</p></div>
<div class="toolbar"><div class="segmented" role="group" aria-label="Simulation scene setting"><button type="button" data-scene="clean" aria-pressed="true">Clean scenes</button><button type="button" data-scene="randomized" aria-pressed="false">Randomized scenes</button></div><p class="note" id="scene-status" role="status">Showing clean scenes · original recordings</p></div>
<div class="video-grid simulation-grid">{''.join(sim_cards)}</div><p class="note">Task success rates are manuscript statistics over 100 episodes per setting, not estimates from the two displayed recordings. Simulation videos retain the source resolution (320 × 240), frame rate (10 fps), and timing.</p>
<noscript><p class="noscript">JavaScript enables sample and scene switching. The initial videos and all paper content remain available. Browse all 34 videos through the <a href="https://github.com/RoboOpus/ACG-WAM/tree/website/assets/videos">video directory</a>.</p></noscript></section>
<section class="section shell" id="citation" aria-labelledby="citation-title"><div class="section-heading"><div><div class="eyebrow">05 / Reference</div><h2 id="citation-title">Citation</h2></div><p>Current manuscript citation. Publication details will be updated when available.</p></div><div class="citation-box"><div class="citation-tools"><span>BibTeX</span><button class="button" id="copy-citation" type="button">Copy citation</button></div><pre><code id="bibtex">{escape(bib)}</code></pre><div class="copy-status" id="copy-status" role="status"></div></div></section>
</main>
<footer><div class="shell"><p>ACG-WAM · World-Action Modeling via Action-Conditioned Geometric Latent Prediction</p><p>Presentation inspired by <a href="https://github.com/AIGeeksGroup/DeltaWAM/tree/website">DeltaWAM</a>. <a href="#top">Back to top ↑</a></p></div></footer>
</body></html>'''
(ROOT/'index.html').write_text(html,encoding='utf8')
print('Built index.html: 10 real videos and 24 simulation videos available through selectors.')
