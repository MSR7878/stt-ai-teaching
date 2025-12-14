"""
Generate annotation taxonomy diagram for Week 3: Data Labeling
Shows different types of annotation tasks across modalities
"""
from graphviz import Digraph

def generate_annotation_taxonomy():
    """Create comprehensive annotation taxonomy diagram."""
    dot = Digraph('Annotation Taxonomy',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white', 'splines': 'ortho'})

    # Root
    dot.node('root', 'Annotation Tasks', shape='box', style='filled',
             fillcolor='lightcoral', fontsize='14', fontname='bold')

    # Main modalities
    dot.node('text', 'Text', shape='box', style='filled', fillcolor='lightblue')
    dot.node('image', 'Image', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('audio', 'Audio', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('video', 'Video', shape='box', style='filled', fillcolor='lightpink')
    dot.node('multimodal', 'Multimodal', shape='box', style='filled', fillcolor='lavender')

    dot.edge('root', 'text')
    dot.edge('root', 'image')
    dot.edge('root', 'audio')
    dot.edge('root', 'video')
    dot.edge('root', 'multimodal')

    # Text tasks
    dot.node('text_cls', 'Classification', shape='ellipse', style='filled', fillcolor='#E3F2FD')
    dot.node('ner', 'NER', shape='ellipse', style='filled', fillcolor='#E3F2FD')
    dot.node('rel_ext', 'Relation\nExtraction', shape='ellipse', style='filled', fillcolor='#E3F2FD')
    dot.node('qa', 'Question\nAnswering', shape='ellipse', style='filled', fillcolor='#E3F2FD')
    dot.node('sentiment', 'Sentiment', shape='ellipse', style='filled', fillcolor='#E3F2FD')
    dot.node('summ', 'Summarization', shape='ellipse', style='filled', fillcolor='#E3F2FD')

    dot.edge('text', 'text_cls')
    dot.edge('text', 'ner')
    dot.edge('text', 'rel_ext')
    dot.edge('text', 'qa')
    dot.edge('text', 'sentiment')
    dot.edge('text', 'summ')

    # Image tasks
    dot.node('img_cls', 'Classification', shape='ellipse', style='filled', fillcolor='#E8F5E9')
    dot.node('det', 'Object\nDetection', shape='ellipse', style='filled', fillcolor='#E8F5E9')
    dot.node('seg_sem', 'Semantic\nSegmentation', shape='ellipse', style='filled', fillcolor='#E8F5E9')
    dot.node('seg_inst', 'Instance\nSegmentation', shape='ellipse', style='filled', fillcolor='#E8F5E9')
    dot.node('keypoints', 'Keypoint\nDetection', shape='ellipse', style='filled', fillcolor='#E8F5E9')

    dot.edge('image', 'img_cls')
    dot.edge('image', 'det')
    dot.edge('image', 'seg_sem')
    dot.edge('image', 'seg_inst')
    dot.edge('image', 'keypoints')

    # Audio tasks
    dot.node('transcribe', 'Transcription', shape='ellipse', style='filled', fillcolor='#FFFDE7')
    dot.node('sound_evt', 'Sound Event\nDetection', shape='ellipse', style='filled', fillcolor='#FFFDE7')
    dot.node('speaker', 'Speaker\nRecognition', shape='ellipse', style='filled', fillcolor='#FFFDE7')
    dot.node('emotion', 'Emotion\nRecognition', shape='ellipse', style='filled', fillcolor='#FFFDE7')

    dot.edge('audio', 'transcribe')
    dot.edge('audio', 'sound_evt')
    dot.edge('audio', 'speaker')
    dot.edge('audio', 'emotion')

    # Video tasks
    dot.node('action', 'Action\nRecognition', shape='ellipse', style='filled', fillcolor='#FCE4EC')
    dot.node('tracking', 'Object\nTracking', shape='ellipse', style='filled', fillcolor='#FCE4EC')
    dot.node('temporal_seg', 'Temporal\nSegmentation', shape='ellipse', style='filled', fillcolor='#FCE4EC')

    dot.edge('video', 'action')
    dot.edge('video', 'tracking')
    dot.edge('video', 'temporal_seg')

    # Multimodal tasks
    dot.node('caption', 'Image\nCaptioning', shape='ellipse', style='filled', fillcolor='#F3E5F5')
    dot.node('vqa', 'Visual QA', shape='ellipse', style='filled', fillcolor='#F3E5F5')

    dot.edge('multimodal', 'caption')
    dot.edge('multimodal', 'vqa')

    # Save diagram
    dot.render('../figures/week03_annotation_taxonomy', format='png', cleanup=True)
    print("Generated: figures/week03_annotation_taxonomy.png")

if __name__ == '__main__':
    print("Generating Week 3 annotation taxonomy diagram...")
    generate_annotation_taxonomy()
    print("Week 3 diagram generated successfully!")
