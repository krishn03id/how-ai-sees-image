from manimlib import *
import numpy as np
from PIL import Image

# ImageMobject is in manimlib's namespace via *, but import explicitly as a safety net.
try:
    from manimlib.mobject.image import ImageMobject
except Exception:
    pass

PHOTO_PATH = "/tmp/digit7_16.png"


def make_digit_image(size=16):
    """A small grayscale '7' -- the canonical 'AI sees an image' example."""
    img = np.zeros((size, size), dtype=np.float32)
    img[3:6, 3:max(13, size)] = 1.0               # top horizontal bar
    for t in np.linspace(0, 1, 60):               # diagonal stroke
        r = 5 + t * (size - 7)
        c = (size - 4) - t * (size - 7)
        ri, ci = int(round(r)), int(round(c))
        ri = max(0, min(size - 1, ri))
        ci = max(0, min(size - 1, ci))
        img[max(0, ri - 1):ri + 2, max(0, ci - 1):ci + 2] = 1.0
    return img


def convolve(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out


EDGE_KERNEL = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)


def make_value_grid(arr, cell=0.3, show_numbers=False, color_fn=None,
                    stroke_color=GREY_D):
    if color_fn is None:
        color_fn = lambda v: interpolate_color(BLACK, WHITE, v)
    h, w = arr.shape
    cells = VGroup()
    for r in range(h):
        for c in range(w):
            v = max(0.0, min(1.0, float(arr[r, c])))
            s = Square(side_length=cell)
            s.set_fill(color_fn(v), 1.0)
            s.set_stroke(stroke_color, 0.5)
            x = (c - (w - 1) / 2.0) * cell
            y = ((h - 1) / 2.0 - r) * cell
            s.move_to([x, y, 0])
            if show_numbers:
                val = 255 if v > 0.5 else 0
                t = Text(str(val), font_size=int(cell * 24),
                         color=(BLACK if v > 0.5 else WHITE))
                t.move_to(s)
                cells.add(VGroup(s, t))
            else:
                cells.add(s)
    return cells


class HowAISees(Scene):
    def construct(self):
        self.beat_title()
        self.beat_image_is_numbers()
        self.beat_channels()
        self.beat_convolution()
        self.beat_layers()
        self.beat_classification()
        self.beat_outro()

    # ---------- 1. title ----------
    def beat_title(self):
        t = Text("How an AI sees an image", font_size=56)
        sub = Text("pixels  ->  features  ->  meaning", font_size=32, color=GREY_B)
        sub.next_to(t, DOWN, buff=0.4)
        self.play(Write(t), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(VGroup(t, sub)))

    # ---------- 2 & 3. image = grid of numbers ----------
    def beat_image_is_numbers(self):
        img = make_digit_image(16)
        Image.fromarray((img * 255).astype(np.uint8)).convert("RGB").save(PHOTO_PATH)

        # part A: the photo vs. the grid
        photo = ImageMobject(PHOTO_PATH)
        photo.set_height(3.0)
        photo.to_edge(LEFT, buff=1.6)
        cap = Text("What you see: the number 7", font_size=28)
        cap.next_to(photo, DOWN, buff=0.4)

        grid = make_value_grid(img, cell=0.26)
        grid.next_to(photo, RIGHT, buff=2.0)
        gcap = Text("What the AI sees: a grid of numbers", font_size=28)
        gcap.next_to(grid, UP, buff=0.4)

        self.play(FadeIn(photo, shift=RIGHT * 0.3), Write(cap))
        self.play(Write(gcap))
        self.play(LaggedStart(*[FadeIn(c, scale=0.5) for c in grid],
                              lag_ratio=0.01, run_time=2.0))
        self.wait(1.0)
        self.play(FadeOut(VGroup(photo, cap, grid, gcap)))

        # part B: zoom -- each cell is a number
        crop = img[2:7, 2:7]
        big = make_value_grid(crop, cell=0.9, show_numbers=True)
        zcap = Text("each pixel = a number   (0 = black  <->  255 = white)",
                    font_size=26)
        zcap.next_to(big, DOWN, buff=0.5)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in big],
                              lag_ratio=0.1, run_time=1.5))
        self.play(Write(zcap))
        self.wait(1.5)
        self.play(FadeOut(VGroup(big, zcap)))

    # ---------- 4. color = 3 channels ----------
    def beat_channels(self):
        img = make_digit_image(10)
        labels = ["R", "G", "B"]
        cols = [RED, GREEN, BLUE]
        grids = VGroup()
        for lab, col in zip(labels, cols):
            g = make_value_grid(img, cell=0.24)
            l = Text(lab, font_size=30, color=col)
            l.next_to(g, DOWN, buff=0.2)
            grids.add(VGroup(g, l))
        grids.arrange(RIGHT, buff=0.9)
        cap = Text("A color image = 3 of these grids (R, G, B).  Grayscale = 1.",
                   font_size=26)
        cap.to_edge(UP, buff=0.5)
        self.play(Write(cap))
        self.play(LaggedStart(*[FadeIn(x, shift=UP * 0.2) for x in grids],
                              lag_ratio=0.2, run_time=1.5))
        self.wait(1.5)
        self.play(FadeOut(VGroup(cap, grids)))

    # ---------- 5. convolution ----------
    def beat_convolution(self):
        img = make_digit_image(12)
        feat = convolve(img, EDGE_KERNEL)
        fmax = max(abs(feat).max(), 1e-6)
        feat_disp = np.abs(feat) / fmax

        cell = 0.28
        inp = make_value_grid(img, cell=cell)
        inp.to_edge(LEFT, buff=1.3)
        fmap = make_value_grid(
            feat_disp, cell=cell,
            color_fn=lambda v: interpolate_color(BLUE_E, YELLOW, v))
        fmap.next_to(inp, RIGHT, buff=2.2)

        cap = Text("A filter slides over the image, computing weighted sums",
                   font_size=24)
        cap.to_edge(UP, buff=0.4)
        cap2 = Text("->  a feature map  (here, vertical edges light up)",
                    font_size=24, color=YELLOW)
        cap2.next_to(cap, DOWN, buff=0.2)

        k = Rectangle(width=cell * 3, height=cell * 3, color=YELLOW,
                      stroke_width=3)
        k.move_to(inp[1 * 12 + 1].get_center())

        self.play(Write(cap))
        self.play(FadeIn(inp), ShowCreation(k))
        tour = [(1, 1), (1, 10), (10, 10), (10, 1), (5, 6)]
        for (r, c) in tour:
            self.play(k.animate.move_to(inp[r * 12 + c].get_center()),
                      run_time=0.4)
        self.play(Write(cap2), FadeIn(fmap))
        self.wait(1.2)
        self.play(FadeOut(VGroup(cap, cap2, inp, fmap, k)))

    # ---------- 6. layers ----------
    def beat_layers(self):
        cap = Text("Stack many layers:  edges  ->  parts  ->  objects",
                   font_size=28)
        cap.to_edge(UP, buff=0.5)

        def corner():
            return VGroup(Line(ORIGIN, RIGHT * 0.5, color=GREEN),
                          Line(ORIGIN, UP * 0.5, color=GREEN))

        edges = VGroup(
            Line(LEFT * 0.5, RIGHT * 0.5, color=BLUE),
            Line(UP * 0.5, DOWN * 0.5, color=BLUE),
            Line(LEFT * 0.4 + UP * 0.4, RIGHT * 0.4 + DOWN * 0.4, color=BLUE),
        ).arrange(RIGHT, buff=0.5)
        parts = VGroup(corner(), corner(), corner()).arrange(RIGHT, buff=0.5)
        obj = Text("7", font_size=64, color=YELLOW)

        l1 = Text("edges", font_size=24, color=BLUE)
        l1.next_to(edges, DOWN, buff=0.3)
        l2 = Text("parts", font_size=24, color=GREEN)
        l2.next_to(parts, DOWN, buff=0.3)
        l3 = Text("object", font_size=24, color=YELLOW)
        l3.next_to(obj, DOWN, buff=0.3)
        g1 = VGroup(edges, l1)
        g2 = VGroup(parts, l2)
        g3 = VGroup(obj, l3)
        row = VGroup(g1, g2, g3).arrange(RIGHT, buff=1.7)
        a1 = Arrow(g1.get_right(), g2.get_left(), buff=0.1)
        a2 = Arrow(g2.get_right(), g3.get_left(), buff=0.1)

        self.play(Write(cap))
        self.play(ShowCreation(g1))
        self.play(ShowCreation(a1), ShowCreation(g2))
        self.play(ShowCreation(a2), ShowCreation(g3))
        self.wait(1.5)
        self.play(FadeOut(VGroup(cap, row, a1, a2)))

    # ---------- 7. classification ----------
    def beat_classification(self):
        probs = [0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.91, 0.01, 0.01]
        cap = Text("Final layer: a probability for each class", font_size=28)
        cap.to_edge(UP, buff=0.5)
        maxh, bw, gap = 2.4, 0.5, 0.18
        baseline_y = -1.4
        bars = VGroup()
        labs = VGroup()
        for i, p in enumerate(probs):
            h = maxh * p
            col = YELLOW if i == 7 else GREY_B
            bar = Rectangle(width=bw, height=h, fill_opacity=0.9, stroke_width=0)
            bar.set_fill(col, 0.9)
            x = (i - 4.5) * (bw + gap)
            bar.move_to([x, baseline_y + h / 2, 0])
            bars.add(bar)
            t = Text(str(i), font_size=22, color=(YELLOW if i == 7 else WHITE))
            t.move_to([x, baseline_y - 0.3, 0])
            labs.add(t)
        axis = Line(LEFT * 3.1, RIGHT * 3.1).shift([0, baseline_y, 0])

        self.play(Write(cap), ShowCreation(axis))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                              lag_ratio=0.05, run_time=2))
        self.play(FadeIn(labs))
        self.wait(0.5)

        verdict = Text("It's a 7!", font_size=56, color=YELLOW)
        verdict.move_to([0, baseline_y + 3.5, 0])
        self.play(Write(verdict))
        self.wait(1.8)
        self.play(FadeOut(VGroup(cap, axis, bars, labs, verdict)))

    # ---------- 8. outro ----------
    def beat_outro(self):
        lines = VGroup(
            Text("An image is just a grid of numbers.", font_size=34),
            Text("Filters turn numbers into features.", font_size=34),
            Text("Layers turn features into meaning.", font_size=34),
        ).arrange(DOWN, buff=0.4)
        self.play(LaggedStart(*[Write(l) for l in lines],
                              lag_ratio=0.6, run_time=3))
        self.wait(2.0)
        self.play(FadeOut(lines))
