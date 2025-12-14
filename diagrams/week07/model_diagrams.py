"""
Generate model development diagrams for Week 7: Model Development
Converts Mermaid diagrams to professional graphviz diagrams
"""
from graphviz import Digraph

def generate_model_lifecycle():
    """Create ML model lifecycle diagram."""
    dot = Digraph('Model Lifecycle', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Define nodes
    dot.node('Data', 'Data Prep', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Feat', 'Feat Eng', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Select', 'Model\nSelection', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Train', 'Training', shape='box', style='filled', fillcolor='lightcoral')
    dot.node('Eval', 'Evaluation', shape='box', style='filled', fillcolor='lightpink')
    dot.node('Good', 'Good\nEnough?', shape='diamond', style='filled', fillcolor='orange')
    dot.node('Deploy', 'Deployment', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Error', 'Error\nAnalysis', shape='box', style='filled', fillcolor='lightyellow')

    # Main flow
    dot.edge('Data', 'Feat')
    dot.edge('Feat', 'Select')
    dot.edge('Select', 'Train')
    dot.edge('Train', 'Eval')
    dot.edge('Eval', 'Good')
    dot.edge('Good', 'Deploy', label='Yes', color='green')
    dot.edge('Good', 'Select', label='No', color='red')

    # Error analysis feedback
    dot.edge('Eval', 'Error')
    dot.edge('Error', 'Data', style='dashed')

    # Save diagram
    dot.render('../figures/week07_model_lifecycle', format='png', cleanup=True)
    print("Generated: figures/week07_model_lifecycle.png")

def generate_bias_variance():
    """Create bias-variance tradeoff diagram."""
    dot = Digraph('Bias-Variance', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Total error node
    dot.node('Total', 'Total Error', shape='box', style='filled', fillcolor='lightcoral')

    # Components
    dot.node('Bias', 'Bias²\n(Underfitting)', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Var', 'Variance\n(Overfitting)', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Irr', 'Irreducible\nError', shape='box', style='filled', fillcolor='lightgray')

    # Decomposition
    dot.edge('Total', 'Bias', label='component')
    dot.edge('Total', 'Var', label='component')
    dot.edge('Total', 'Irr', label='component')

    # Save diagram
    dot.render('../figures/week07_bias_variance', format='png', cleanup=True)
    print("Generated: figures/week07_bias_variance.png")

def generate_automl_ensemble():
    """Create AutoML ensemble diagram."""
    dot = Digraph('AutoML Ensemble', graph_attr={'rankdir': 'LR', 'bgcolor': 'white'})

    # Data
    dot.node('Data', 'Training\nData', shape='cylinder', style='filled', fillcolor='lightgray')

    # Base models
    dot.node('RF', 'Random\nForest', shape='box', style='filled', fillcolor='lightblue')
    dot.node('CB', 'CatBoost', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('NN', 'Neural Net', shape='box', style='filled', fillcolor='lightgreen')

    # Ensemble
    dot.node('Ensemble', 'Weighted\nEnsemble\n(Stacking)', shape='box', style='filled', fillcolor='lightcoral')

    # Prediction
    dot.node('Pred', 'Prediction', shape='ellipse', style='filled', fillcolor='lightpink')

    # Flow
    dot.edge('Data', 'RF')
    dot.edge('Data', 'CB')
    dot.edge('Data', 'NN')

    dot.edge('RF', 'Ensemble', label='pred 1')
    dot.edge('CB', 'Ensemble', label='pred 2')
    dot.edge('NN', 'Ensemble', label='pred 3')

    dot.edge('Ensemble', 'Pred')

    # Save diagram
    dot.render('../figures/week07_automl_ensemble', format='png', cleanup=True)
    print("Generated: figures/week07_automl_ensemble.png")

def generate_transfer_learning():
    """Create transfer learning diagram."""
    dot = Digraph('Transfer Learning', graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Pre-trained backbone
    dot.node('Backbone', 'Pre-trained Backbone\n(ImageNet weights)\nFROZEN',
             shape='box', style='filled,dashed', fillcolor='lightgray')

    # New head
    dot.node('Head', 'New Classification Head\nTRAINABLE',
             shape='box', style='filled', fillcolor='lightgreen')

    # Connection
    dot.edge('Backbone', 'Head', label='features')

    # Add legend
    dot.node('Legend', 'Feature Extraction Mode:\n* Backbone: Frozen (reuse learned features)\n* Head: Trainable (learn task-specific classifier)',
             shape='note', style='filled', fillcolor='lightyellow')

    # Save diagram
    dot.render('../figures/week07_transfer_learning', format='png', cleanup=True)
    print("Generated: figures/week07_transfer_learning.png")

if __name__ == '__main__':
    print("Generating Week 7 diagrams...")
    generate_model_lifecycle()
    generate_bias_variance()
    generate_automl_ensemble()
    generate_transfer_learning()
    print("All Week 7 diagrams generated successfully!")
