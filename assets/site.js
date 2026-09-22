'use strict';

function setVideo(video, src, poster, label) {
  const wasPlaying = !video.paused;
  video.pause();
  video.querySelector('source').src = src;
  video.querySelector('a').href = src;
  video.poster = poster;
  if (label) video.setAttribute('aria-label', label);
  video.load();
  if (wasPlaying) video.play().catch(() => {});
}

document.querySelectorAll('[data-real-select]').forEach(select => {
  select.addEventListener('change', () => {
    const option = select.selectedOptions[0];
    const card = select.closest('[data-real-card]');
    setVideo(card.querySelector('video'), option.value, option.dataset.poster,
      `${card.querySelector('h3').textContent} — ${option.textContent}`);
  });
});

let scene = 'clean';
function updateSimulation(card) {
  const episode = card.querySelector('[data-episode]').value;
  const prefix = `${card.dataset.base}/demo_${scene}/episode${episode}`;
  const label = scene === 'clean' ? 'Clean' : 'Randomized';
  setVideo(card.querySelector('video'), `${prefix}.mp4`, `${prefix}.jpg`,
    `${card.dataset.title} — ${label}, episode ${episode}`);
  card.querySelector('[data-scene-label]').textContent = label;
  card.querySelector('[data-score]').textContent = `${label} success rate: ${card.dataset[scene]}%`;
}
document.querySelectorAll('[data-scene]').forEach(button => {
  button.addEventListener('click', () => {
    if (scene === button.dataset.scene) return;
    scene = button.dataset.scene;
    document.querySelectorAll('[data-scene]').forEach(item => {
      item.setAttribute('aria-pressed', String(item.dataset.scene === scene));
    });
    document.querySelectorAll('[data-sim-card]').forEach(updateSimulation);
    document.getElementById('scene-status').textContent = `Showing ${scene} scenes · original recordings`;
  });
});
document.querySelectorAll('[data-episode]').forEach(select => {
  select.addEventListener('change', () => updateSimulation(select.closest('[data-sim-card]')));
});

// Pause off-screen players without restarting videos paused by the reader.
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) entry.target.pause();
    });
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
