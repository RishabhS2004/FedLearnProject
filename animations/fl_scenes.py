"""
RadioFed Manim Animations — 3Blue1Brown Style

Dark navy background, clean geometry, contrasting warm/cool tones.
Render: manim -qh animations/fl_scenes.py <SceneName> --media_dir static/manim_out
"""

from manim import *
import numpy as np

# ── 3B1B-style palette ──
BG       = "#1c1c2e"   # deep navy
BLUE     = "#58c4dd"   # bright cyan-blue
YELLOW   = "#e8c445"   # warm gold
GREEN    = "#83c167"   # muted green
RED      = "#e85d75"   # soft red
WHITE    = "#ece6e2"   # warm white
GREY     = "#888888"
DBLUE    = "#3b6ea5"
PURPLE   = "#b189e8"


class FederatedLearningFlow(Scene):
    """Complete FL round: partition → extract → train → upload → filter → aggregate → distribute."""

    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = Text("Federated Learning Round", font_size=42, color=WHITE, weight=BOLD)
        sub = Text("RadioFed — Byzantine-Resilient AMC", font_size=22, color=GREY)
        sub.next_to(title, DOWN, buff=0.3)
        self.play(Write(title, run_time=1.2), FadeIn(sub, shift=UP*0.2))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # ── Server ──
        server = self._node("Server", BLUE, UP*2.2)
        self.play(FadeIn(server, scale=0.8))

        # ── Clients ──
        cx = [-4.5, -1.5, 1.5, 4.5]
        colors = [BLUE, DBLUE, GREEN, RED]
        labels = ["Client 1", "Client 2", "Client 3", "Client 4"]
        clients = [self._node(l, c, RIGHT*x + DOWN*2) for l, c, x in zip(labels, colors, cx)]
        self.play(*[FadeIn(c, shift=DOWN*0.4) for c in clients], lag_ratio=0.12)
        self.wait(0.5)

        # ── Phase 1: Local Training ──
        ph = self._phase("1. Local Training", YELLOW)
        self.play(FadeIn(ph))

        rings = []
        for cl, col in zip(clients, colors):
            r = Circle(radius=0.25, color=col, stroke_width=2.5).move_to(cl)
            rings.append(r)
        self.play(*[Create(r) for r in rings])
        self.play(*[Rotate(r, 2*PI) for r in rings], run_time=1.5)
        # Accuracy labels
        accs = ["91%", "87%", "93%", "41%"]
        acc_t = []
        for cl, a, col in zip(clients, accs, colors):
            tc = GREEN if a != "41%" else RED
            t = Text(a, font_size=18, color=tc).next_to(cl, UP, buff=0.12)
            acc_t.append(t)
        self.play(*[FadeIn(a) for a in acc_t], *[FadeOut(r) for r in rings])
        self.wait(0.8)

        # ── Phase 2: Upload ──
        self.play(FadeOut(ph))
        ph2 = self._phase("2. Upload Models", BLUE)
        self.play(FadeIn(ph2))
        for cl, col in zip(clients, colors):
            d = Dot(radius=0.06, color=col).move_to(cl.get_top())
            line = Line(cl.get_top(), server.get_bottom(), color=col, stroke_width=1.5, buff=0.25)
            self.play(Create(line, run_time=0.25))
            self.play(MoveAlongPath(d, line, run_time=0.35))
            self.play(FadeOut(d), FadeOut(line), run_time=0.15)
        self.play(*[FadeOut(a) for a in acc_t])
        self.wait(0.5)

        # ── Phase 3: Byzantine Detection ──
        self.play(FadeOut(ph2))
        ph3 = self._phase("3. Byzantine Detection", YELLOW)
        self.play(FadeIn(ph3))

        shield = RegularPolygon(6, color=YELLOW, fill_opacity=0.12, stroke_width=2).scale(0.3)
        shield.move_to(server.get_center() + UP*0.55)
        self.play(FadeIn(shield, scale=0.5))

        # Scan
        scan = Line(LEFT*5, RIGHT*5, color=YELLOW, stroke_width=1.5, stroke_opacity=0.5)
        scan.move_to(UP*1.5)
        self.play(scan.animate.move_to(DOWN*2.8), run_time=1.2, rate_func=linear)
        self.play(FadeOut(scan))

        # Mark Byzantine
        x_mark = Cross(clients[3], stroke_color=RED, stroke_width=4)
        rej = Text("REJECTED", font_size=14, color=RED).next_to(clients[3], DOWN, buff=0.12)
        checks = [Text("✓", font_size=22, color=GREEN).next_to(clients[i], DOWN, buff=0.12) for i in range(3)]
        self.play(Create(x_mark), Write(rej))
        self.play(*[Write(c) for c in checks])
        self.wait(1)
        self.play(FadeOut(shield))

        # ── Phase 4: Aggregation ──
        self.play(FadeOut(ph3))
        ph4 = self._phase("4. Aggregate", GREEN)
        self.play(FadeIn(ph4))

        glow = Circle(radius=0.5, color=GREEN, fill_opacity=0.15, stroke_width=2).move_to(server)
        self.play(Create(glow))
        self.play(glow.animate.scale(1.6).set_opacity(0), run_time=0.8)
        self.remove(glow)

        result = Text("Global Model  94.2%", font_size=20, color=GREEN)
        result.next_to(server, RIGHT, buff=0.35)
        self.play(Write(result))
        self.wait(0.8)

        # ── Phase 5: Distribute ──
        self.play(FadeOut(ph4))
        ph5 = self._phase("5. Distribute", BLUE)
        self.play(FadeIn(ph5))

        for i in range(3):
            arr = Arrow(server.get_bottom(), clients[i].get_top(), color=GREEN, stroke_width=2, buff=0.25, max_tip_length_to_length_ratio=0.12)
            self.play(Create(arr, run_time=0.25))
            self.play(FadeOut(arr, run_time=0.15))
        self.wait(1)

        # ── End ──
        self.play(*[FadeOut(m) for m in self.mobjects])
        end = Text("Round Complete", font_size=36, color=WHITE, weight=BOLD)
        end2 = Text("3/4 clients aggregated  ·  Byzantine client rejected", font_size=20, color=GREY)
        end2.next_to(end, DOWN, buff=0.3)
        self.play(Write(end), FadeIn(end2, shift=UP*0.2))
        self.wait(2)

    def _node(self, label, color, pos):
        r = RoundedRectangle(corner_radius=0.12, width=1.7, height=0.85,
                             color=color, fill_opacity=0.08, stroke_width=1.8)
        t = Text(label, font_size=15, color=color)
        t.move_to(r)
        g = VGroup(r, t).move_to(pos)
        return g

    def _phase(self, text, color):
        return Text(text, font_size=24, color=color, weight=BOLD).to_edge(UP, buff=0.25)


class ByzantineDetection(Scene):
    """Feature-space visualization of Byzantine detection via Krum."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Byzantine Fault Detection", font_size=36, color=WHITE, weight=BOLD)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP, buff=0.25).scale(0.75))

        # Axes
        ax = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], x_length=5.5, y_length=5.5,
                  axis_config={"color": GREY, "stroke_width": 0.8}).shift(LEFT*0.5)
        self.play(Create(ax, run_time=0.8))

        # Honest clusters
        np.random.seed(42)
        honest_c = [(0.5, 0.7, BLUE), (-0.4, 0.4, DBLUE), (0.1, -0.4, GREEN)]
        all_dots = VGroup()
        for cx, cy, col in honest_c:
            pts = np.random.randn(12, 2)*0.35 + [cx, cy]
            for p in pts:
                d = Dot(ax.c2p(p[0], p[1]), radius=0.04, color=col, fill_opacity=0.7)
                all_dots.add(d)

        # Byzantine — far away
        byz_pts = np.random.randn(12, 2)*0.25 + [2.0, -2.0]
        byz_dots = VGroup()
        for p in byz_pts:
            d = Dot(ax.c2p(p[0], p[1]), radius=0.04, color=RED, fill_opacity=0.7)
            byz_dots.add(d)

        self.play(FadeIn(all_dots, run_time=0.8))

        # Labels
        for (cx, cy, col), name in zip(honest_c, ["Client 1", "Client 2", "Client 3"]):
            lb = Text(name, font_size=12, color=col).move_to(ax.c2p(cx+0.6, cy+0.3))
            self.play(FadeIn(lb, run_time=0.3))

        self.wait(0.3)
        byz_lb = Text("Client 4", font_size=12, color=RED).move_to(ax.c2p(2.3, -1.5))
        self.play(FadeIn(byz_dots), Write(byz_lb))

        # Median
        med = Dot(ax.c2p(0.07, 0.23), radius=0.09, color=YELLOW, fill_opacity=0.8)
        med_lb = Text("Median", font_size=11, color=YELLOW).next_to(med, UP, buff=0.12)
        self.play(FadeIn(med), Write(med_lb))

        # Detection circle
        circ = Circle(radius=ax.c2p(1.5, 0)[0]-ax.c2p(0, 0)[0],
                      color=YELLOW, stroke_width=1.5, stroke_opacity=0.5)
        circ.move_to(med)
        self.play(Create(circ))
        self.wait(0.5)

        # Reject
        xm = Cross(byz_dots, stroke_color=RED, stroke_width=3)
        rej = Text("REJECTED", font_size=16, color=RED).next_to(byz_dots, DOWN, buff=0.15)
        self.play(Create(xm), Write(rej))

        # Trust sidebar
        sidebar = VGroup()
        st = Text("Trust Scores", font_size=16, color=WHITE).to_edge(RIGHT, buff=0.4).shift(UP*1.5)
        sidebar.add(st)
        scores = [("Client 1", 0.72, GREEN), ("Client 2", 0.68, GREEN),
                  ("Client 3", 0.65, GREEN), ("Client 4", 0.08, RED)]
        for i, (n, s, c) in enumerate(scores):
            bg = Rectangle(width=1.8, height=0.18, color=GREY, fill_opacity=0.08, stroke_width=0.5)
            fill = Rectangle(width=1.8*s, height=0.18, color=c, fill_opacity=0.5, stroke_width=0)
            fill.align_to(bg, LEFT)
            lb = Text(f"{n}: {s:.2f}", font_size=10, color=c)
            lb.next_to(bg, LEFT, buff=0.08)
            g = VGroup(bg, fill, lb)
            g.next_to(st, DOWN, buff=0.25+i*0.42)
            g.align_to(st, LEFT)
            sidebar.add(g)

        self.play(FadeIn(sidebar, shift=LEFT*0.3))
        self.wait(2.5)


class TrustEvolution(Scene):
    """Trust score curves over 10 rounds — 3B1B style."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Trust Score Evolution", font_size=36, color=WHITE, weight=BOLD)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP, buff=0.25).scale(0.75))

        ax = Axes(
            x_range=[0, 10, 1], y_range=[0, 1, 0.2],
            x_length=9.5, y_length=5,
            axis_config={"color": GREY, "stroke_width": 0.8,
                         "include_numbers": True, "font_size": 20},
        ).shift(DOWN*0.2)
        xl = Text("Aggregation Round", font_size=14, color=GREY).next_to(ax.x_axis, DOWN, buff=0.25)
        yl = Text("Trust", font_size=14, color=GREY).next_to(ax.y_axis, LEFT, buff=0.2).rotate(PI/2)
        self.play(Create(ax, run_time=0.8), Write(xl), Write(yl))

        # Threshold
        thr = DashedLine(ax.c2p(0, 0.3), ax.c2p(10, 0.3),
                         color=RED, stroke_width=1.2, dash_length=0.08)
        thr_lb = Text("Threshold", font_size=10, color=RED).next_to(thr, RIGHT, buff=0.08)
        self.play(Create(thr), Write(thr_lb))

        # Data
        rounds = list(range(11))
        curves = [
            ([0.5,0.6,0.65,0.7,0.75,0.78,0.82,0.85,0.87,0.89,0.91], BLUE, "Client 1"),
            ([0.5,0.55,0.6,0.63,0.68,0.72,0.75,0.78,0.8,0.83,0.85], DBLUE, "Client 2"),
            ([0.5,0.58,0.62,0.67,0.7,0.74,0.77,0.8,0.82,0.85,0.87], GREEN, "Client 3"),
            ([0.5,0.4,0.3,0.22,0.16,0.12,0.09,0.07,0.05,0.04,0.03], RED, "Client 4"),
        ]

        legend = VGroup()
        for data, color, name in curves:
            graph = ax.plot_line_graph(x_values=rounds, y_values=data,
                                       line_color=color, add_vertex_dots=True,
                                       vertex_dot_radius=0.035, stroke_width=2.5)
            self.play(Create(graph, run_time=1.2))

            dot = Dot(radius=0.05, color=color)
            lb = Text(name, font_size=11, color=color)
            lb.next_to(dot, RIGHT, buff=0.08)
            legend.add(VGroup(dot, lb))

        legend.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        legend.to_corner(UR, buff=0.5).shift(DOWN*0.5)
        bg = SurroundingRectangle(legend, color=GREY, fill_opacity=0.06, buff=0.12, corner_radius=0.08, stroke_width=0.5)
        self.play(FadeIn(bg), FadeIn(legend))

        # Highlight rejection point
        rp = Dot(ax.c2p(2, 0.3), radius=0.1, color=RED, fill_opacity=0.4)
        rl = Text("Rejected at round 2", font_size=12, color=RED).next_to(rp, DOWN, buff=0.15)
        self.play(FadeIn(rp, scale=0.5), Write(rl))
        self.wait(2.5)


class SignalClassification(Scene):
    """AM vs FM waveforms and feature extraction flow."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Automatic Modulation Classification", font_size=34, color=WHITE, weight=BOLD)
        self.play(Write(title, run_time=1))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP, buff=0.25).scale(0.75))

        # AM
        am_ax = Axes(x_range=[0,4,1], y_range=[-2,2,1], x_length=4.5, y_length=2.2,
                     axis_config={"color":GREY,"stroke_width":0.6}).shift(LEFT*3+UP*0.5)
        am_lb = Text("AM Signal", font_size=18, color=BLUE).next_to(am_ax, UP, buff=0.15)
        t = np.linspace(0,4,400)
        am_y = (1+0.7*np.sin(2*np.pi*0.5*t))*np.sin(2*np.pi*5*t)
        am_g = am_ax.plot_line_graph(x_values=t, y_values=am_y,
                                      line_color=BLUE, add_vertex_dots=False, stroke_width=1.8)

        # FM
        fm_ax = Axes(x_range=[0,4,1], y_range=[-2,2,1], x_length=4.5, y_length=2.2,
                     axis_config={"color":GREY,"stroke_width":0.6}).shift(RIGHT*3+UP*0.5)
        fm_lb = Text("FM Signal", font_size=18, color=YELLOW).next_to(fm_ax, UP, buff=0.15)
        fm_y = np.sin(2*np.pi*5*t+3*np.sin(2*np.pi*0.5*t))
        fm_g = fm_ax.plot_line_graph(x_values=t, y_values=fm_y,
                                      line_color=YELLOW, add_vertex_dots=False, stroke_width=1.8)

        self.play(Create(am_ax), Write(am_lb), Create(am_g),
                  Create(fm_ax), Write(fm_lb), Create(fm_g), run_time=1.8)
        self.wait(0.8)

        # Feature extraction
        arr_down = Arrow(ORIGIN+DOWN*0.5, ORIGIN+DOWN*1.5, color=GREY, stroke_width=1.5,
                         max_tip_length_to_length_ratio=0.15)
        fe_lb = Text("Feature Extraction  (16D)", font_size=20, color=PURPLE).shift(DOWN*1.8)
        self.play(Create(arr_down), Write(fe_lb))

        feats = ["Amp μ", "Amp σ²", "Amp Skew", "Amp Kurt",
                 "Freq μ", "Freq σ²", "Freq Skew", "Freq Kurt"]
        feat_grp = VGroup()
        for i, f in enumerate(feats):
            t = Text(f, font_size=12, color=WHITE)
            feat_grp.add(t)
        feat_grp.arrange_in_grid(2, 4, buff=0.3).shift(DOWN*2.7)
        self.play(FadeIn(feat_grp, shift=UP*0.2, lag_ratio=0.06))
        self.wait(0.8)

        # Classification
        arr2 = Arrow(ORIGIN+DOWN*3.2, ORIGIN+DOWN*3.8, color=GREY, stroke_width=1.5,
                     max_tip_length_to_length_ratio=0.15)
        cls_lb = Text("KNN · DT · RF · GB · SVM · LR · NB · MLP", font_size=16, color=GREEN)
        cls_lb.shift(DOWN*3.5)
        result = Text("→  AM  or  FM", font_size=20, color=WHITE, weight=BOLD).next_to(cls_lb, DOWN, buff=0.3)
        self.play(Create(arr2), Write(cls_lb))
        self.play(Write(result))
        self.wait(2)


class AggregationProcess(Scene):
    """Data-centric federated aggregation flow."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Data-Centric Federated Aggregation", font_size=34, color=WHITE, weight=BOLD)
        self.play(Write(title, run_time=1))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP, buff=0.25).scale(0.75))

        # Client data boxes
        colors = [BLUE, DBLUE, GREEN]
        boxes = []
        for i, (col, cnt) in enumerate(zip(colors, [2400, 2400, 2400])):
            r = RoundedRectangle(corner_radius=0.1, width=1.8, height=1,
                                 color=col, fill_opacity=0.08, stroke_width=1.5)
            t = Text(f"Client {i+1}\n{cnt} samples", font_size=13, color=col)
            t.move_to(r)
            g = VGroup(r, t).shift(LEFT*4 + UP*(1.2-i*1.5))
            boxes.append(g)

        self.play(*[FadeIn(b, shift=RIGHT*0.3) for b in boxes])

        # Arrow → Merge
        arr1 = Arrow(LEFT*2.5, LEFT*0.5, color=GREY, stroke_width=1.5)
        merge_t = Text("Merge", font_size=16, color=YELLOW).next_to(arr1, UP, buff=0.08)
        self.play(Create(arr1), Write(merge_t))

        # Combined
        comb = RoundedRectangle(corner_radius=0.1, width=2.2, height=1.8,
                                color=YELLOW, fill_opacity=0.08, stroke_width=1.5).shift(RIGHT*1)
        comb_t = Text("Combined\n7,200 samples", font_size=14, color=YELLOW).move_to(comb)
        self.play(FadeIn(comb), Write(comb_t))

        # Arrow → Retrain
        arr2 = Arrow(RIGHT*2.5, RIGHT*4, color=GREY, stroke_width=1.5)
        retr_t = Text("Retrain", font_size=16, color=GREEN).next_to(arr2, UP, buff=0.08)
        self.play(Create(arr2), Write(retr_t))

        # Global model
        glob = RoundedRectangle(corner_radius=0.12, width=2, height=1.3,
                                color=GREEN, fill_opacity=0.1, stroke_width=2).shift(RIGHT*5.2)
        glob_t = Text("Global\nModel\n94.2%", font_size=14, color=GREEN, weight=BOLD).move_to(glob)
        self.play(FadeIn(glob), Write(glob_t))

        # Glow
        glow = glob.copy().set_stroke(GREEN, width=10, opacity=0.2)
        self.play(FadeIn(glow))
        self.wait(2)
