"""
Generate visual examples for different annotation tasks (simplified version)
Week 3: Data Labeling & Annotation
"""
from graphviz import Digraph

def generate_iou_visualization():
    """Create IoU metric visualization using boxes."""
    dot = Digraph('IoU Metric',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Good IoU
    dot.node('good_title', 'Good IoU = 0.75', shape='box', style='filled',
             fillcolor='lightgreen', fontsize='14', fontname='bold')

    dot.node('good_gt', 'Ground Truth Box\n[100, 100, 100, 100]',
             shape='box', style='filled', fillcolor='lightblue', width='2', height='1.5')
    dot.node('good_pred', 'Predicted Box\n[110, 110, 100, 100]',
             shape='box', style='filled,dashed', fillcolor='lightcoral', width='2', height='1.5')
    dot.node('good_result', 'Intersection = 8100\nUnion = 10900\nIoU = 0.74 ✓',
             shape='box', style='filled', fillcolor='lightyellow')

    # Poor IoU
    dot.node('poor_title', 'Poor IoU = 0.15', shape='box', style='filled',
             fillcolor='lightcoral', fontsize='14', fontname='bold')

    dot.node('poor_gt', 'Ground Truth Box\n[50, 50, 100, 100]',
             shape='box', style='filled', fillcolor='lightblue', width='2', height='1.5')
    dot.node('poor_pred', 'Predicted Box\n[120, 120, 100, 100]',
             shape='box', style='filled,dashed', fillcolor='lightcoral', width='2', height='1.5')
    dot.node('poor_result', 'Intersection = 900\nUnion = 19100\nIoU = 0.05 ✗',
             shape='box', style='filled', fillcolor='lightyellow')

    # Edges
    dot.edge('good_title', 'good_gt', style='invis')
    dot.edge('good_gt', 'good_pred', label='Overlap', color='green')
    dot.edge('good_pred', 'good_result', style='invis')

    dot.edge('poor_title', 'poor_gt', style='invis')
    dot.edge('poor_gt', 'poor_pred', label='Little Overlap', color='red')
    dot.edge('poor_pred', 'poor_result', style='invis')

    dot.render('../figures/week03_iou_visualization', format='png', cleanup=True)
    print("Generated: figures/week03_iou_visualization.png")

def generate_audio_annotation_example():
    """Create audio annotation timeline."""
    dot = Digraph('Audio Annotation',
                  graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Timeline
    dot.node('start', '0s', shape='plaintext')
    dot.node('evt1', 'door_slam\n[2.3-3.1s]', shape='box', style='filled',
             fillcolor='lightcoral', width='1.5')
    dot.node('t5', '5s', shape='plaintext')
    dot.node('evt2', 'dog_bark\n[5.0-8.2s]', shape='box', style='filled',
             fillcolor='lightyellow', width='2')
    dot.node('t10', '10s', shape='plaintext')
    dot.node('evt3', 'glass_break\n[10.5-11.0s]', shape='box', style='filled',
             fillcolor='lightgreen')
    dot.node('t15', '15s', shape='plaintext')
    dot.node('evt4', 'music\n[15.0-45.0s]', shape='box', style='filled',
             fillcolor='lightblue', width='3')
    dot.node('end', '50s', shape='plaintext')

    # Timeline flow
    dot.edge('start', 'evt1', label='silent')
    dot.edge('evt1', 't5', label='silent')
    dot.edge('t5', 'evt2')
    dot.edge('evt2', 't10', label='silent')
    dot.edge('t10', 'evt3')
    dot.edge('evt3', 't15', label='silent')
    dot.edge('t15', 'evt4')
    dot.edge('evt4', 'end')

    # Title
    dot.node('title', 'Audio Event Detection Timeline\nFile: home_audio.wav',
             shape='box', style='filled', fillcolor='lightgray', fontsize='12')
    dot.edge('title', 'start', style='invis')

    dot.render('../figures/week03_audio_annotation_example', format='png', cleanup=True)
    print("Generated: figures/week03_audio_annotation_example.png")

def generate_video_tracking_example():
    """Create video tracking across frames."""
    dot = Digraph('Video Tracking',
                  graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Frame 0
    dot.node('f0_title', 'Frame 0 (t=0.00s)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('f0_car', 'Car (ID:1)\nBBox: [100, 200, 50, 80]',
             shape='box', style='filled', fillcolor='#FFB6C1')
    dot.node('f0_person', 'Person (ID:2)\nBBox: [800, 150, 120, 350]',
             shape='box', style='filled', fillcolor='#87CEEB')

    # Frame 1
    dot.node('f1_title', 'Frame 1 (t=0.03s)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('f1_car', 'Car (ID:1)\nBBox: [105, 202, 50, 80]',
             shape='box', style='filled', fillcolor='#FFB6C1')
    dot.node('f1_person', 'Person (ID:2)\nBBox: [805, 152, 120, 350]',
             shape='box', style='filled', fillcolor='#87CEEB')

    # Frame 2
    dot.node('f2_title', 'Frame 2 (t=0.07s)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('f2_car', 'Car (ID:1)\nBBox: [110, 204, 50, 80]',
             shape='box', style='filled', fillcolor='#FFB6C1')
    dot.node('f2_person', 'Person (ID:2)\nBBox: [810, 154, 120, 350]',
             shape='box', style='filled', fillcolor='#87CEEB')

    # Connections within frames
    dot.edge('f0_title', 'f0_car', style='invis')
    dot.edge('f0_title', 'f0_person', style='invis')
    dot.edge('f1_title', 'f1_car', style='invis')
    dot.edge('f1_title', 'f1_person', style='invis')
    dot.edge('f2_title', 'f2_car', style='invis')
    dot.edge('f2_title', 'f2_person', style='invis')

    # Tracking arrows
    dot.edge('f0_car', 'f1_car', label='Track ID:1', color='red', penwidth='2')
    dot.edge('f1_car', 'f2_car', label='Track ID:1', color='red', penwidth='2')
    dot.edge('f0_person', 'f1_person', label='Track ID:2', color='blue', penwidth='2')
    dot.edge('f1_person', 'f2_person', label='Track ID:2', color='blue', penwidth='2')

    dot.render('../figures/week03_video_tracking_example', format='png', cleanup=True)
    print("Generated: figures/week03_video_tracking_example.png")

def generate_sentiment_annotation_example():
    """Create sentiment annotation example."""
    dot = Digraph('Sentiment Annotation',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Text
    dot.node('text', 'Text: "Great camera but poor battery life and decent price"',
             shape='box', style='filled', fillcolor='lightgray', fontsize='12')

    # Aspects
    dot.node('camera', 'Aspect: camera\nSentiment: POSITIVE',
             shape='box', style='filled', fillcolor='lightgreen')
    dot.node('battery', 'Aspect: battery\nSentiment: NEGATIVE',
             shape='box', style='filled', fillcolor='lightcoral')
    dot.node('price', 'Aspect: price\nSentiment: NEUTRAL',
             shape='box', style='filled', fillcolor='lightyellow')

    # Overall
    dot.node('overall', 'Overall Sentiment: MIXED\n(Contains both + and -)',
             shape='box', style='filled', fillcolor='lightblue', fontsize='11')

    dot.edge('text', 'camera', label='"Great camera"')
    dot.edge('text', 'battery', label='"poor battery life"')
    dot.edge('text', 'price', label='"decent price"')

    dot.edge('camera', 'overall', style='invis')
    dot.edge('battery', 'overall', style='invis')
    dot.edge('price', 'overall', style='invis')

    dot.render('../figures/week03_sentiment_example', format='png', cleanup=True)
    print("Generated: figures/week03_sentiment_example.png")

def generate_qa_annotation_example():
    """Create QA annotation example."""
    dot = Digraph('QA Annotation',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Context
    dot.node('context', 'Context:\n"The Apollo program landed 12 astronauts\non the Moon between 1969 and 1972."',
             shape='box', style='filled', fillcolor='lightgray', width='5')

    # Question
    dot.node('question', 'Question:\n"When did the Apollo program land astronauts?"',
             shape='ellipse', style='filled', fillcolor='lightyellow', width='4')

    # Answer
    dot.node('answer', 'Answer Span:\n"between 1969 and 1972"',
             shape='box', style='filled', fillcolor='lightgreen', width='3')

    # Annotation details
    dot.node('details', 'answer_start: 54\nanswer_text: "between 1969 and 1972"',
             shape='note', style='filled', fillcolor='lightblue')

    dot.edge('context', 'question', label='extract from')
    dot.edge('question', 'answer', label='answer is')
    dot.edge('answer', 'details', style='dashed')

    dot.render('../figures/week03_qa_example', format='png', cleanup=True)
    print("Generated: figures/week03_qa_example.png")

if __name__ == '__main__':
    print("Generating simplified Week 3 annotation example diagrams...")

    # Generate remaining diagrams
    generate_iou_visualization()
    generate_audio_annotation_example()
    generate_video_tracking_example()
    generate_sentiment_annotation_example()
    generate_qa_annotation_example()

    print("\nAll Week 3 annotation example diagrams generated successfully!")
