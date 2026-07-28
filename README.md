# How an AI sees an image

A short ManimGL animation showing how a neural network turns an image into a
prediction: **pixels → numbers → feature maps (convolutions) → layers →
class probabilities**.

The example image is a small grayscale "7" — the canonical "AI recognizing a
digit" case.

## Render locally

```bash
pip install -r requirements.txt
manimgl scene.py HowAISees -w -m
```

## Render on GitHub Actions

`.github/workflows/render.yml` renders the video **headlessly** on every push
to `main` (and via `workflow_dispatch`). ManimGL uses a live OpenGL/GLFW
window, so on CI it runs under `xvfb` with software Mesa (llvmpipe).

The finished `.mp4` is uploaded as a build artifact **and** attached to a
`latest` release:

https://github.com/krishn03id/how-ai-sees-image/releases/latest

## Credit

Built with [ManimGL](https://github.com/3b1b/manim), 3Blue1Brown's animation
engine.
