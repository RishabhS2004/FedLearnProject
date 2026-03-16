"""
RadioFed Manim — Small-card versions (rendered 640x360)
Larger text, thicker lines, fewer elements — crisp at small display sizes.
"""

from manim import *
import numpy as np

config.pixel_width = 640
config.pixel_height = 360
config.frame_rate = 30
config.frame_width = 10
config.frame_height = 5.625

BG = "#1c1c2e"
BLUE = "#58c4dd"
YELLOW = "#e8c445"
GREEN = "#83c167"
RED = "#e85d75"
WHITE = "#ece6e2"
GREY = "#888888"
DBLUE = "#3b6ea5"
PURPLE = "#b189e8"


class SignalClassification(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Signal Classification", font_size=32, color=WHITE, weight=BOLD)
        self.play(Write(title, run_time=0.8))
        self.wait(0.3)
        self.play(title.animate.to_edge(UP, buff=0.2).scale(0.8))

        # AM
        am_ax = Axes(x_range=[0,4,1], y_range=[-2,2,1], x_length=3.8, y_length=1.8,
                     axis_config={"color":GREY,"stroke_width":0.8}).shift(LEFT*2.5+UP*0.2)
        am_lb = Text("AM", font_size=22, color=BLUE, weight=BOLD).next_to(am_ax, UP, buff=0.1)
        t = np.linspace(0,4,300)
        am_y = (1+0.7*np.sin(2*np.pi*0.5*t))*np.sin(2*np.pi*5*t)
        am_g = am_ax.plot_line_graph(x_values=t, y_values=am_y,
                                      line_color=BLUE, add_vertex_dots=False, stroke_width=2)

        # FM
        fm_ax = Axes(x_range=[0,4,1], y_range=[-2,2,1], x_length=3.8, y_length=1.8,
                     axis_config={"color":GREY,"stroke_width":0.8}).shift(RIGHT*2.5+UP*0.2)
        fm_lb = Text("FM", font_size=22, color=YELLOW, weight=BOLD).next_to(fm_ax, UP, buff=0.1)
        fm_y = np.sin(2*np.pi*5*t+3*np.sin(2*np.pi*0.5*t))
        fm_g = fm_ax.plot_line_graph(x_values=t, y_values=fm_y,
                                      line_color=YELLOW, add_vertex_dots=False, stroke_width=2)

        self.play(Create(am_ax), Write(am_lb), Create(am_g),
                  Create(fm_ax), Write(fm_lb), Create(fm_g), run_time=1.2)
        self.wait(0.5)

        # Arrow + classification
        arr = Arrow(ORIGIN+DOWN*0.8, ORIGIN+DOWN*1.5, color=GREY, stroke_width=2)
        cls_lb = Text("16D Features → ML Classifier → AM or FM",
                      font_size=18, color=GREEN, weight=BOLD).shift(DOWN*2)
        self.play(Create(arr), Write(cls_lb, run_time=0.8))
        self.wait(2)


class ByzantineDetection(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Byzantine Detection", font_size=32, color=WHITE, weight=BOLD)
        self.play(Write(title, run_time=0.8))
        self.wait(0.3)
        self.play(title.animate.to_edge(UP, buff=0.2).scale(0.8))

        # Simplified axes
        ax = Axes(x_range=[-3,3,1], y_range=[-3,3,1], x_length=4, y_length=4,
                  axis_config={"color":GREY,"stroke_width":0.6}).shift(LEFT*1)
        self.play(Create(ax, run_time=0.5))

        # Honest clusters — bigger dots
        np.random.seed(42)
        honest = [(0.5,0.7,BLUE,"C1"),(-0.4,0.4,DBLUE,"C2"),(0.1,-0.4,GREEN,"C3")]
        for cx,cy,col,name in honest:
            pts = np.random.randn(8,2)*0.35+[cx,cy]
            dots = VGroup(*[Dot(ax.c2p(p[0],p[1]),radius=0.06,color=col,fill_opacity=0.8) for p in pts])
            lb = Text(name,font_size=16,color=col,weight=BOLD).move_to(ax.c2p(cx+0.7,cy+0.3))
            self.play(FadeIn(dots,run_time=0.3),FadeIn(lb,run_time=0.3))

        # Byzantine
        byz_pts = np.random.randn(8,2)*0.25+[2,-2]
        byz_dots = VGroup(*[Dot(ax.c2p(p[0],p[1]),radius=0.06,color=RED,fill_opacity=0.8) for p in byz_pts])
        byz_lb = Text("C4",font_size=16,color=RED,weight=BOLD).move_to(ax.c2p(2.3,-1.5))
        self.play(FadeIn(byz_dots),Write(byz_lb))

        # Detection circle
        med = Dot(ax.c2p(0.07,0.23),radius=0.1,color=YELLOW,fill_opacity=0.9)
        circ = Circle(radius=ax.c2p(1.5,0)[0]-ax.c2p(0,0)[0],
                      color=YELLOW,stroke_width=2,stroke_opacity=0.6).move_to(med)
        self.play(FadeIn(med),Create(circ))

        # Reject
        xm = Cross(byz_dots,stroke_color=RED,stroke_width=5)
        rej = Text("REJECTED",font_size=20,color=RED,weight=BOLD).next_to(byz_dots,DOWN,buff=0.15)
        self.play(Create(xm),Write(rej))

        # Trust sidebar
        trust_title = Text("Trust",font_size=18,color=WHITE,weight=BOLD).shift(RIGHT*3.5+UP*1.2)
        self.play(Write(trust_title))
        scores = [("C1",0.72,GREEN),("C2",0.68,GREEN),("C3",0.65,GREEN),("C4",0.08,RED)]
        for i,(n,s,c) in enumerate(scores):
            bg = Rectangle(width=1.5,height=0.2,color=GREY,fill_opacity=0.1,stroke_width=0.5)
            fill = Rectangle(width=1.5*s,height=0.2,color=c,fill_opacity=0.6,stroke_width=0)
            fill.align_to(bg,LEFT)
            lb = Text(f"{n} {s:.2f}",font_size=12,color=c,weight=BOLD)
            lb.next_to(bg,LEFT,buff=0.06)
            g = VGroup(bg,fill,lb).next_to(trust_title,DOWN,buff=0.2+i*0.35).align_to(trust_title,LEFT)
            self.play(FadeIn(g,run_time=0.2))

        self.wait(2)


class AggregationProcess(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Data-Centric Aggregation", font_size=32, color=WHITE, weight=BOLD)
        self.play(Write(title, run_time=0.8))
        self.wait(0.3)
        self.play(title.animate.to_edge(UP, buff=0.2).scale(0.8))

        # Client boxes
        colors = [BLUE, DBLUE, GREEN]
        boxes = []
        for i,(col,cnt) in enumerate(zip(colors,[2400,2400,2400])):
            r = RoundedRectangle(corner_radius=0.1,width=1.6,height=0.8,
                                 color=col,fill_opacity=0.1,stroke_width=2)
            t = Text(f"Client {i+1}\n{cnt}",font_size=14,color=col,weight=BOLD)
            t.move_to(r)
            g = VGroup(r,t).shift(LEFT*3.5+UP*(0.9-i*1.1))
            boxes.append(g)

        self.play(*[FadeIn(b,shift=RIGHT*0.3) for b in boxes])

        arr1 = Arrow(LEFT*2,LEFT*0.3,color=GREY,stroke_width=2)
        merge = Text("Merge",font_size=18,color=YELLOW,weight=BOLD).next_to(arr1,UP,buff=0.06)
        self.play(Create(arr1),Write(merge))

        comb = RoundedRectangle(corner_radius=0.1,width=1.8,height=1.3,
                                color=YELLOW,fill_opacity=0.1,stroke_width=2).shift(RIGHT*0.8)
        comb_t = Text("7,200\nsamples",font_size=16,color=YELLOW,weight=BOLD).move_to(comb)
        self.play(FadeIn(comb),Write(comb_t))

        arr2 = Arrow(RIGHT*2,RIGHT*3.5,color=GREY,stroke_width=2)
        retr = Text("Retrain",font_size=18,color=GREEN,weight=BOLD).next_to(arr2,UP,buff=0.06)
        self.play(Create(arr2),Write(retr))

        glob = RoundedRectangle(corner_radius=0.12,width=1.6,height=1,
                                color=GREEN,fill_opacity=0.12,stroke_width=2.5).shift(RIGHT*4.3)
        glob_t = Text("Global\n94.2%",font_size=18,color=GREEN,weight=BOLD).move_to(glob)
        self.play(FadeIn(glob),Write(glob_t))

        glow = glob.copy().set_stroke(GREEN,width=10,opacity=0.25)
        self.play(FadeIn(glow))
        self.wait(2)


class TrustEvolution(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Trust Score Evolution", font_size=32, color=WHITE, weight=BOLD)
        self.play(Write(title, run_time=0.8))
        self.wait(0.3)
        self.play(title.animate.to_edge(UP, buff=0.2).scale(0.8))

        ax = Axes(
            x_range=[0,10,2], y_range=[0,1,0.2],
            x_length=8, y_length=3.5,
            axis_config={"color":GREY,"stroke_width":0.8,
                         "include_numbers":True,"font_size":18},
        ).shift(DOWN*0.3)
        xl = Text("Round",font_size=14,color=GREY).next_to(ax.x_axis,DOWN,buff=0.2)
        yl = Text("Trust",font_size=14,color=GREY).next_to(ax.y_axis,LEFT,buff=0.15).rotate(PI/2)
        self.play(Create(ax,run_time=0.6),Write(xl),Write(yl))

        # Threshold
        thr = DashedLine(ax.c2p(0,0.3),ax.c2p(10,0.3),color=RED,stroke_width=1.5,dash_length=0.08)
        self.play(Create(thr))

        rounds = list(range(11))
        curves = [
            ([0.5,0.6,0.65,0.7,0.75,0.78,0.82,0.85,0.87,0.89,0.91],BLUE,"C1"),
            ([0.5,0.55,0.6,0.63,0.68,0.72,0.75,0.78,0.8,0.83,0.85],DBLUE,"C2"),
            ([0.5,0.58,0.62,0.67,0.7,0.74,0.77,0.8,0.82,0.85,0.87],GREEN,"C3"),
            ([0.5,0.4,0.3,0.22,0.16,0.12,0.09,0.07,0.05,0.04,0.03],RED,"C4"),
        ]

        legend = VGroup()
        for data,color,name in curves:
            graph = ax.plot_line_graph(x_values=rounds,y_values=data,
                                       line_color=color,add_vertex_dots=True,
                                       vertex_dot_radius=0.04,stroke_width=3)
            self.play(Create(graph,run_time=1))
            dot = Dot(radius=0.06,color=color)
            lb = Text(name,font_size=14,color=color,weight=BOLD)
            lb.next_to(dot,RIGHT,buff=0.06)
            legend.add(VGroup(dot,lb))

        legend.arrange(DOWN,buff=0.1,aligned_edge=LEFT)
        legend.to_corner(UR,buff=0.4).shift(DOWN*0.3)
        bg = SurroundingRectangle(legend,color=GREY,fill_opacity=0.08,buff=0.1,corner_radius=0.06,stroke_width=0.5)
        self.play(FadeIn(bg),FadeIn(legend))

        rej = Text("C4 rejected",font_size=14,color=RED,weight=BOLD).move_to(ax.c2p(5,0.05))
        self.play(Write(rej))
        self.wait(2)
