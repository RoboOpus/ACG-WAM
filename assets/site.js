'use strict';

let scene = 'clean';
let episode = '0';
function updateSimulation() {
  document.querySelectorAll('[data-sim-card]').forEach(card => {
    const video = card.querySelector('video');
    const wasPlaying = !video.paused;
    const prefix = `${card.dataset.base}/demo_${scene}/episode${episode}`;
    video.pause();
    video.querySelector('source').src = `${prefix}.mp4`;
    video.querySelector('a').href = `${prefix}.mp4`;
    video.poster = `${prefix}.jpg`;
    video.setAttribute('aria-label', `${card.dataset.title} — ${scene}, demonstration ${Number(episode) + 1}`);
    video.load();
    if (wasPlaying) video.play().catch(() => {});
  });
  document.getElementById('scene-status').textContent = `${scene === 'clean' ? 'Clean' : 'Randomized'} scenes, demonstration ${Number(episode) + 1}`;
}
document.querySelectorAll('[data-scene]').forEach(button => {
  button.addEventListener('click', () => {
    if (scene === button.dataset.scene) return;
    scene = button.dataset.scene;
    document.querySelectorAll('[data-scene]').forEach(item => {
      item.setAttribute('aria-pressed', String(item.dataset.scene === scene));
    });
    updateSimulation();
  });
});
document.querySelectorAll('[data-episode]').forEach(button => {
  button.addEventListener('click', () => {
    if (episode === button.dataset.episode) return;
    episode = button.dataset.episode;
    document.querySelectorAll('[data-episode]').forEach(item => {
      item.setAttribute('aria-pressed', String(item.dataset.episode === episode));
    });
    updateSimulation();
  });
});

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (!entry.isIntersecting) entry.target.pause(); });
  }, { threshold: 0.05 });
  document.querySelectorAll('video').forEach(video => observer.observe(video));
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) document.querySelectorAll('video').forEach(video => video.pause());
});

document.getElementById('copy-citation').addEventListener('click', async () => {
  const code = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = 'Citation copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C (or ⌘C) to copy.';
  }
});
