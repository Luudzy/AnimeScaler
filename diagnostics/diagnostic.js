const input = document.querySelector('#file');
const video = document.querySelector('#video');
let url;
input.addEventListener('change', () => {
  if (url) URL.revokeObjectURL(url);
  const file = input.files?.[0];
  if (!file) return;
  url = URL.createObjectURL(file);
  video.src = url;
  video.play().catch(() => {});
});
