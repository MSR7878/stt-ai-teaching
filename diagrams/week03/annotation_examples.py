"""
Generate visual examples for different annotation tasks
Week 3: Data Labeling & Annotation
"""
from graphviz import Digraph
import os

def generate_object_detection_example():
    """Create object detection bounding box visualization."""
    dot = Digraph('Object Detection',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'},
                  node_attr={'shape': 'plaintext'})

    # Create HTML-like table to represent image with bounding boxes
    html = '''<
    <TABLE BORDER="2" CELLBORDER="0" CELLSPACING="0" CELLPADDING="10" BGCOLOR="lightgray">
        <TR><TD COLSPAN="3" BGCOLOR="lightblue"><B>Image: street_scene.jpg (1920x1080)</B></TD></TR>
        <TR>
            <TD COLSPAN="3" HEIGHT="300" WIDTH="400" BGCOLOR="white">
                <TABLE BORDER="3" CELLBORDER="1" CELLSPACING="20">
                    <TR>
                        <TD BGCOLOR="red" ALPHA="0.3" WIDTH="100" HEIGHT="60">
                            <FONT COLOR="red"><B>car</B></FONT><BR/>
                            <FONT POINT-SIZE="10">[100, 200, 400, 500]</FONT>
                        </TD>
                        <TD WIDTH="50"></TD>
                        <TD BGCOLOR="blue" ALPHA="0.3" WIDTH="40" HEIGHT="100">
                            <FONT COLOR="blue"><B>person</B></FONT><BR/>
                            <FONT POINT-SIZE="10">[800, 150, 120, 350]</FONT>
                        </TD>
                    </TR>
                    <TR>
                        <TD BGCOLOR="red" ALPHA="0.3" WIDTH="80" HEIGHT="50">
                            <FONT COLOR="red"><B>car</B></FONT><BR/>
                            <FONT POINT-SIZE="10">[50, 600, 350, 450]</FONT>
                        </TD>
                        <TD COLSPAN="2"></TD>
                    </TR>
                </TABLE>
            </TD>
        </TR>
        <TR>
            <TD BGCOLOR="lightcoral"><B>Bounding Box Format:</B><BR/>[x, y, width, height]</TD>
            <TD BGCOLOR="lightgreen"><B>Classes:</B><BR/>car (red), person (blue)</TD>
            <TD BGCOLOR="lightyellow"><B>Total Objects:</B><BR/>3 (2 cars, 1 person)</TD>
        </TR>
    </TABLE>
    >'''

    dot.node('detection', html)
    dot.render('../figures/week03_object_detection_example', format='png', cleanup=True)
    print("Generated: figures/week03_object_detection_example.png")

def generate_segmentation_comparison():
    """Create semantic vs instance segmentation comparison."""
    dot = Digraph('Segmentation Types',
                  graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Semantic segmentation
    sem_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD COLSPAN="3" BGCOLOR="lightblue"><B>Semantic Segmentation</B></TD></TR>
        <TR><TD COLSPAN="3"><B>All instances share same label</B></TD></TR>
        <TR>
            <TD BGCOLOR="lightgreen" WIDTH="80" HEIGHT="80">Person<BR/>(class 1)</TD>
            <TD BGCOLOR="lightgreen" WIDTH="80" HEIGHT="80">Person<BR/>(class 1)</TD>
            <TD BGCOLOR="lightcoral" WIDTH="80" HEIGHT="80">Car<BR/>(class 2)</TD>
        </TR>
        <TR>
            <TD BGCOLOR="gray" COLSPAN="3">Road (class 3)</TD>
        </TR>
        <TR><TD COLSPAN="3" BGCOLOR="lightyellow">3 classes total</TD></TR>
    </TABLE>
    >'''

    # Instance segmentation
    inst_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD COLSPAN="3" BGCOLOR="lightblue"><B>Instance Segmentation</B></TD></TR>
        <TR><TD COLSPAN="3"><B>Each instance has unique ID</B></TD></TR>
        <TR>
            <TD BGCOLOR="#90EE90" WIDTH="80" HEIGHT="80">Person<BR/>(ID: 1)</TD>
            <TD BGCOLOR="#00FF00" WIDTH="80" HEIGHT="80">Person<BR/>(ID: 2)</TD>
            <TD BGCOLOR="lightcoral" WIDTH="80" HEIGHT="80">Car<BR/>(ID: 3)</TD>
        </TR>
        <TR>
            <TD BGCOLOR="gray" COLSPAN="3">Road (background)</TD>
        </TR>
        <TR><TD COLSPAN="3" BGCOLOR="lightyellow">3 instances total</TD></TR>
    </TABLE>
    >'''

    dot.node('semantic', sem_html, shape='plaintext')
    dot.node('instance', inst_html, shape='plaintext')

    dot.render('../figures/week03_segmentation_comparison', format='png', cleanup=True)
    print("Generated: figures/week03_segmentation_comparison.png")

def generate_keypoint_example():
    """Create human pose keypoint visualization."""
    dot = Digraph('Keypoint Detection',
                  graph_attr={'bgcolor': 'white', 'rankdir': 'TB'})

    # Define keypoints as nodes
    keypoints = {
        'nose': (5, 1),
        'left_eye': (4, 1),
        'right_eye': (6, 1),
        'left_ear': (3, 1),
        'right_ear': (7, 1),
        'left_shoulder': (3, 3),
        'right_shoulder': (7, 3),
        'left_elbow': (2, 5),
        'right_elbow': (8, 5),
        'left_wrist': (1, 7),
        'right_wrist': (9, 7),
        'left_hip': (3.5, 6),
        'right_hip': (6.5, 6),
        'left_knee': (3.5, 9),
        'right_knee': (6.5, 9),
        'left_ankle': (3.5, 12),
        'right_ankle': (6.5, 12)
    }

    # Create nodes
    for name, (x, y) in keypoints.items():
        visible = 1 if 'right_shoulder' not in name else 0  # Make right shoulder occluded
        color = 'green' if visible else 'red'
        dot.node(name, f'{name.replace("_", " ").title()}\n{"✓ visible" if visible else "✗ occluded"}',
                shape='circle', style='filled', fillcolor=color, fontcolor='white' if visible else 'white',
                width='1.5', fontsize='9', pos=f'{x},{y}!')

    # Create skeleton connections
    skeleton = [
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')
    ]

    for kp1, kp2 in skeleton:
        dot.edge(kp1, kp2, dir='none', penwidth='2', color='blue')

    dot.render('../figures/week03_keypoint_example', format='png', cleanup=True)
    print("Generated: figures/week03_keypoint_example.png")

def generate_ner_example():
    """Create NER text annotation visualization."""
    dot = Digraph('NER Example',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'},
                  node_attr={'shape': 'plaintext'})

    html = '''<
    <TABLE BORDER="2" CELLBORDER="0" CELLSPACING="5" CELLPADDING="8">
        <TR><TD COLSPAN="10" BGCOLOR="lightblue"><B>Named Entity Recognition (NER)</B></TD></TR>
        <TR><TD COLSPAN="10" BGCOLOR="white" ALIGN="LEFT">
            <FONT POINT-SIZE="12"><B>Text:</B> "Apple CEO Tim Cook announced iPhone 15 in Cupertino on Sep 12."</FONT>
        </TD></TR>
        <TR><TD COLSPAN="10" HEIGHT="20"></TD></TR>
        <TR>
            <TD BGCOLOR="lightcoral" BORDER="3"><B>Apple</B><BR/>[0:5]</TD>
            <TD>CEO</TD>
            <TD BGCOLOR="lightgreen" BORDER="3"><B>Tim Cook</B><BR/>[10:18]</TD>
            <TD>announced</TD>
            <TD BGCOLOR="lightyellow" BORDER="3"><B>iPhone 15</B><BR/>[29:38]</TD>
            <TD>in</TD>
            <TD BGCOLOR="lightblue" BORDER="3"><B>Cupertino</B><BR/>[42:51]</TD>
            <TD>on</TD>
            <TD BGCOLOR="lavender" BORDER="3"><B>Sep 12</B><BR/>[55:61]</TD>
            <TD>.</TD>
        </TR>
        <TR>
            <TD BGCOLOR="lightcoral"><FONT POINT-SIZE="10">ORG</FONT></TD>
            <TD></TD>
            <TD BGCOLOR="lightgreen"><FONT POINT-SIZE="10">PERSON</FONT></TD>
            <TD></TD>
            <TD BGCOLOR="lightyellow"><FONT POINT-SIZE="10">PRODUCT</FONT></TD>
            <TD></TD>
            <TD BGCOLOR="lightblue"><FONT POINT-SIZE="10">LOCATION</FONT></TD>
            <TD></TD>
            <TD BGCOLOR="lavender"><FONT POINT-SIZE="10">DATE</FONT></TD>
            <TD></TD>
        </TR>
    </TABLE>
    >'''

    dot.node('ner', html)
    dot.render('../figures/week03_ner_example', format='png', cleanup=True)
    print("Generated: figures/week03_ner_example.png")

def generate_iou_visualization():
    """Create IoU metric visualization."""
    dot = Digraph('IoU Metric',
                  graph_attr={'rankdir': 'LR', 'bgcolor': 'white'},
                  node_attr={'shape': 'plaintext'})

    # Good IoU example
    good_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10">
        <TR><TD COLSPAN="4" BGCOLOR="lightgreen"><B>Good IoU = 0.75</B></TD></TR>
        <TR>
            <TD ROWSPAN="4" BGCOLOR="blue" ALPHA="0.5" WIDTH="60" HEIGHT="60">
                <FONT COLOR="blue"><B>Ground<BR/>Truth</B></FONT>
            </TD>
            <TD ROWSPAN="3" COLSPAN="2" BGCOLOR="purple" WIDTH="60" HEIGHT="50">
                <FONT COLOR="white"><B>Intersection</B></FONT>
            </TD>
            <TD ROWSPAN="3" BGCOLOR="red" ALPHA="0.5" WIDTH="30">
                <FONT COLOR="red"><B>Pred</B></FONT>
            </TD>
        </TR>
        <TR></TR>
        <TR></TR>
        <TR>
            <TD COLSPAN="3" HEIGHT="10"></TD>
        </TR>
        <TR><TD COLSPAN="4" BGCOLOR="lightyellow">
            IoU = Intersection / Union<BR/>
            = 60 / 80 = 0.75 (Good!)
        </TD></TR>
    </TABLE>
    >'''

    # Poor IoU example
    poor_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10">
        <TR><TD COLSPAN="5" BGCOLOR="lightcoral"><B>Poor IoU = 0.25</B></TD></TR>
        <TR>
            <TD ROWSPAN="3" BGCOLOR="blue" ALPHA="0.5" WIDTH="60" HEIGHT="60">
                <FONT COLOR="blue"><B>Ground<BR/>Truth</B></FONT>
            </TD>
            <TD ROWSPAN="2" BGCOLOR="purple" WIDTH="30" HEIGHT="40">
                <FONT COLOR="white"><B>Int</B></FONT>
            </TD>
            <TD ROWSPAN="3" COLSPAN="2" BGCOLOR="red" ALPHA="0.5" WIDTH="60" HEIGHT="60">
                <FONT COLOR="red"><B>Prediction</B></FONT>
            </TD>
        </TR>
        <TR></TR>
        <TR>
            <TD HEIGHT="20"></TD>
        </TR>
        <TR><TD COLSPAN="4" BGCOLOR="lightyellow">
            IoU = Intersection / Union<BR/>
            = 20 / 100 = 0.20 (Poor!)
        </TD></TR>
    </TABLE>
    >'''

    dot.node('good', good_html)
    dot.node('poor', poor_html)

    dot.render('../figures/week03_iou_visualization', format='png', cleanup=True)
    print("Generated: figures/week03_iou_visualization.png")

def generate_audio_annotation_example():
    """Create audio waveform annotation visualization."""
    dot = Digraph('Audio Annotation',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'},
                  node_attr={'shape': 'plaintext'})

    html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD COLSPAN="12" BGCOLOR="lightblue"><B>Audio Event Detection</B></TD></TR>
        <TR><TD COLSPAN="12" BGCOLOR="white"><B>File: home_audio.wav (Duration: 50 seconds)</B></TD></TR>
        <TR><TD COLSPAN="12" HEIGHT="10" BGCOLOR="gray">
            <FONT POINT-SIZE="8">~~~~~~~~ Waveform ~~~~~~~~</FONT>
        </TD></TR>
        <TR>
            <TD COLSPAN="2" BGCOLOR="white">0s</TD>
            <TD COLSPAN="1" BGCOLOR="lightcoral"><B>door_slam</B><BR/>[2.3-3.1]</TD>
            <TD BGCOLOR="white">5s</TD>
            <TD COLSPAN="3" BGCOLOR="lightyellow"><B>dog_bark</B><BR/>[5.0-8.2]</TD>
            <TD BGCOLOR="white">10s</TD>
            <TD BGCOLOR="lightgreen"><B>glass_break</B><BR/>[10.5-11.0]</TD>
            <TD BGCOLOR="white">15s</TD>
            <TD COLSPAN="2" BGCOLOR="lightblue"><B>music</B><BR/>[15.0-45.0]</TD>
        </TR>
        <TR>
            <TD COLSPAN="12" HEIGHT="10" BGCOLOR="gray">
                <FONT POINT-SIZE="8">Time axis (seconds) →</FONT>
            </TD>
        </TR>
        <TR><TD COLSPAN="12" BGCOLOR="lightyellow">
            <B>Annotation:</B> 4 events labeled with start/end times
        </TD></TR>
    </TABLE>
    >'''

    dot.node('audio', html)
    dot.render('../figures/week03_audio_annotation_example', format='png', cleanup=True)
    print("Generated: figures/week03_audio_annotation_example.png")

def generate_video_tracking_example():
    """Create video object tracking visualization."""
    dot = Digraph('Video Tracking',
                  graph_attr={'rankdir': 'LR', 'bgcolor': 'white'},
                  node_attr={'shape': 'plaintext'})

    # Frame 0
    frame0_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD COLSPAN="3" BGCOLOR="lightblue"><B>Frame 0</B></TD></TR>
        <TR>
            <TD BGCOLOR="red" WIDTH="50" HEIGHT="40">
                <FONT COLOR="white"><B>ID:1</B><BR/>car</FONT>
            </TD>
            <TD WIDTH="30"></TD>
            <TD BGCOLOR="blue" WIDTH="30" HEIGHT="60">
                <FONT COLOR="white"><B>ID:2</B><BR/>person</FONT>
            </TD>
        </TR>
        <TR><TD COLSPAN="3">t = 0.00s</TD></TR>
    </TABLE>
    >'''

    # Frame 1
    frame1_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD COLSPAN="3" BGCOLOR="lightblue"><B>Frame 1</B></TD></TR>
        <TR>
            <TD WIDTH="20"></TD>
            <TD BGCOLOR="red" WIDTH="50" HEIGHT="40">
                <FONT COLOR="white"><B>ID:1</B><BR/>car</FONT>
            </TD>
            <TD BGCOLOR="blue" WIDTH="30" HEIGHT="60">
                <FONT COLOR="white"><B>ID:2</B><BR/>person</FONT>
            </TD>
        </TR>
        <TR><TD COLSPAN="3">t = 0.03s</TD></TR>
    </TABLE>
    >'''

    # Frame 2
    frame2_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD COLSPAN="3" BGCOLOR="lightblue"><B>Frame 2</B></TD></TR>
        <TR>
            <TD WIDTH="30"></TD>
            <TD BGCOLOR="red" WIDTH="50" HEIGHT="40">
                <FONT COLOR="white"><B>ID:1</B><BR/>car</FONT>
            </TD>
            <TD BGCOLOR="blue" WIDTH="30" HEIGHT="60">
                <FONT COLOR="white"><B>ID:2</B><BR/>person</FONT>
            </TD>
        </TR>
        <TR><TD COLSPAN="3">t = 0.07s</TD></TR>
    </TABLE>
    >'''

    dot.node('f0', frame0_html)
    dot.node('f1', frame1_html)
    dot.node('f2', frame2_html)

    dot.edge('f0', 'f1', label='tracking →', color='green', penwidth='2')
    dot.edge('f1', 'f2', label='tracking →', color='green', penwidth='2')

    # Add legend
    legend_html = '''<
    <TABLE BORDER="2" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5">
        <TR><TD BGCOLOR="lightyellow"><B>Key Points:</B></TD></TR>
        <TR><TD>• Same ID across frames</TD></TR>
        <TR><TD>• Position changes tracked</TD></TR>
        <TR><TD>• Bounding boxes follow objects</TD></TR>
    </TABLE>
    >'''

    dot.node('legend', legend_html)

    dot.render('../figures/week03_video_tracking_example', format='png', cleanup=True)
    print("Generated: figures/week03_video_tracking_example.png")

def generate_sentiment_annotation_example():
    """Create aspect-based sentiment analysis visualization."""
    dot = Digraph('Sentiment Annotation',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'},
                  node_attr={'shape': 'plaintext'})

    html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="7" BGCOLOR="lightblue"><B>Aspect-Based Sentiment Analysis</B></TD></TR>
        <TR><TD COLSPAN="7" BGCOLOR="white" ALIGN="LEFT">
            <FONT POINT-SIZE="12"><B>Text:</B> "Great camera but poor battery life and decent price"</FONT>
        </TD></TR>
        <TR><TD COLSPAN="7" HEIGHT="10"></TD></TR>
        <TR>
            <TD BGCOLOR="lightgreen" BORDER="3" COLSPAN="2">
                <B>Great camera</B>
            </TD>
            <TD>but</TD>
            <TD BGCOLOR="lightcoral" BORDER="3" COLSPAN="2">
                <B>poor battery life</B>
            </TD>
            <TD>and</TD>
            <TD BGCOLOR="lightyellow" BORDER="3">
                <B>decent price</B>
            </TD>
        </TR>
        <TR>
            <TD COLSPAN="2" BGCOLOR="lightgreen">
                Aspect: camera<BR/>
                Sentiment: POSITIVE
            </TD>
            <TD></TD>
            <TD COLSPAN="2" BGCOLOR="lightcoral">
                Aspect: battery<BR/>
                Sentiment: NEGATIVE
            </TD>
            <TD></TD>
            <TD BGCOLOR="lightyellow">
                Aspect: price<BR/>
                Sentiment: NEUTRAL
            </TD>
        </TR>
        <TR><TD COLSPAN="7" HEIGHT="10"></TD></TR>
        <TR><TD COLSPAN="7" BGCOLOR="lightgray">
            <B>Overall Sentiment:</B> MIXED (has both positive and negative aspects)
        </TD></TR>
    </TABLE>
    >'''

    dot.node('sentiment', html)
    dot.render('../figures/week03_sentiment_example', format='png', cleanup=True)
    print("Generated: figures/week03_sentiment_example.png")

def generate_qa_annotation_example():
    """Create question answering annotation visualization."""
    dot = Digraph('QA Annotation',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'},
                  node_attr={'shape': 'plaintext'})

    html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
        <TR><TD COLSPAN="2" BGCOLOR="lightblue"><B>Question Answering (Extractive)</B></TD></TR>
        <TR><TD COLSPAN="2" BGCOLOR="white" ALIGN="LEFT">
            <B>Context:</B><BR/>
            "The Apollo program landed 12 astronauts on the Moon between 1969 and 1972."
        </TD></TR>
        <TR><TD COLSPAN="2" HEIGHT="10"></TD></TR>
        <TR><TD BGCOLOR="lightyellow" COLSPAN="2">
            <B>Question:</B> "When did the Apollo program land astronauts?"
        </TD></TR>
        <TR><TD COLSPAN="2" HEIGHT="10"></TD></TR>
        <TR><TD COLSPAN="2" BGCOLOR="white" ALIGN="LEFT">
            The Apollo program landed 12 astronauts on the Moon
            <FONT BGCOLOR="lightgreen" COLOR="white"><B>[between 1969 and 1972]</B></FONT>.
        </TD></TR>
        <TR><TD COLSPAN="2" HEIGHT="10"></TD></TR>
        <TR>
            <TD BGCOLOR="lightgreen"><B>Answer Text:</B><BR/>"between 1969 and 1972"</TD>
            <TD BGCOLOR="lightcoral"><B>Answer Start:</B><BR/>Character position: 54</TD>
        </TR>
        <TR><TD COLSPAN="2" BGCOLOR="lightyellow">
            <B>Annotation Type:</B> Extractive (answer is a span in the context)
        </TD></TR>
    </TABLE>
    >'''

    dot.node('qa', html)
    dot.render('../figures/week03_qa_example', format='png', cleanup=True)
    print("Generated: figures/week03_qa_example.png")

if __name__ == '__main__':
    print("Generating Week 3 annotation example diagrams...")

    # Generate all example diagrams
    generate_object_detection_example()
    generate_segmentation_comparison()
    generate_keypoint_example()
    generate_ner_example()
    generate_iou_visualization()
    generate_audio_annotation_example()
    generate_video_tracking_example()
    generate_sentiment_annotation_example()
    generate_qa_annotation_example()

    print("\nAll Week 3 annotation example diagrams generated successfully!")
    print("Total diagrams: 9")
