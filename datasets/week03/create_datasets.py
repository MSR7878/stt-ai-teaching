"""
Create sample datasets for Label Studio annotation tasks.
Run: uv run create_datasets.py
"""

import json
from pathlib import Path

# Output directory
DATASETS_DIR = Path(__file__).parent


def create_text_classification_dataset():
    """Movie reviews for sentiment classification (positive/negative/neutral)."""
    (DATASETS_DIR / "text_classification").mkdir(exist_ok=True)
    reviews = [
        # Positive
        {"text": "Absolutely loved this movie! The acting was superb and the plot kept me engaged throughout.", "id": 1},
        {"text": "A masterpiece of modern cinema. Every scene was beautifully crafted.", "id": 2},
        {"text": "Best film I've seen this year. Highly recommend to everyone!", "id": 3},
        {"text": "The special effects were stunning and the story was heartwarming.", "id": 4},
        {"text": "Incredible performances by the entire cast. A must-watch!", "id": 5},
        {"text": "This movie exceeded all my expectations. Brilliant storytelling.", "id": 6},
        {"text": "A delightful experience from start to finish. Pure entertainment!", "id": 7},
        {"text": "The director outdid themselves with this one. Visually spectacular.", "id": 8},
        # Negative
        {"text": "Complete waste of time. The plot made no sense whatsoever.", "id": 9},
        {"text": "Terrible acting and a predictable storyline. Very disappointed.", "id": 10},
        {"text": "I walked out halfway through. Couldn't bear to watch anymore.", "id": 11},
        {"text": "The worst movie I've seen in years. Save your money.", "id": 12},
        {"text": "Boring, slow, and completely forgettable. Not worth watching.", "id": 13},
        {"text": "The script was awful and the characters were unlikeable.", "id": 14},
        {"text": "A complete disaster. How did this even get made?", "id": 15},
        {"text": "Two hours of my life I'll never get back. Truly awful.", "id": 16},
        # Neutral/Mixed
        {"text": "It was okay. Nothing special but not terrible either.", "id": 17},
        {"text": "Some good moments but overall pretty average.", "id": 18},
        {"text": "The visuals were great but the story was lacking.", "id": 19},
        {"text": "Decent performances but the pacing was off.", "id": 20},
        {"text": "Had potential but didn't quite deliver.", "id": 21},
        {"text": "A few laughs here and there but mostly forgettable.", "id": 22},
        {"text": "Good cinematography, mediocre everything else.", "id": 23},
        {"text": "Not bad, not great. Just somewhere in the middle.", "id": 24},
        # Ambiguous (for discussion)
        {"text": "Well, that was... something. I'm not sure what I just watched.", "id": 25},
        {"text": "My friend loved it but I found it confusing.", "id": 26},
        {"text": "Great first half, disappointing ending.", "id": 27},
        {"text": "Interesting concept but poor execution.", "id": 28},
        {"text": "The kind of movie you either love or hate.", "id": 29},
        {"text": "Better than expected but still not great.", "id": 30},
    ]

    output_path = DATASETS_DIR / "text_classification" / "movie_reviews.json"
    with open(output_path, "w") as f:
        json.dump(reviews, f, indent=2)
    print(f"Created: {output_path} ({len(reviews)} samples)")


def create_ner_dataset():
    """Sentences for Named Entity Recognition (PERSON, ORG, LOCATION, DATE, PRODUCT)."""
    (DATASETS_DIR / "ner").mkdir(exist_ok=True)
    sentences = [
        {"text": "Apple CEO Tim Cook announced the new iPhone 15 at their headquarters in Cupertino on September 12.", "id": 1},
        {"text": "Elon Musk's Tesla delivered record numbers of Model Y vehicles in Shanghai last quarter.", "id": 2},
        {"text": "Microsoft founder Bill Gates visited the World Health Organization in Geneva yesterday.", "id": 3},
        {"text": "Dr. Sarah Chen from Stanford University published groundbreaking research on Monday.", "id": 4},
        {"text": "Amazon Web Services opened a new data center in Mumbai, India in March 2024.", "id": 5},
        {"text": "The Beatles performed their last concert at Apple Corps headquarters in London.", "id": 6},
        {"text": "NASA astronaut Neil Armstrong landed on the Moon on July 20, 1969.", "id": 7},
        {"text": "Sundar Pichai announced Google's new Pixel 8 phone at the Made by Google event.", "id": 8},
        {"text": "Professor James Wilson from MIT will speak at the conference in Boston next week.", "id": 9},
        {"text": "Netflix CEO Reed Hastings stepped down from his position in January 2023.", "id": 10},
        {"text": "The United Nations headquarters in New York hosted the climate summit.", "id": 11},
        {"text": "Mark Zuckerberg renamed Facebook to Meta in October 2021.", "id": 12},
        {"text": "Toyota announced the new Camry model will be manufactured in Kentucky.", "id": 13},
        {"text": "Dr. Anthony Fauci addressed the National Institutes of Health in Washington D.C.", "id": 14},
        {"text": "Samsung released the Galaxy S24 series at their Unpacked event in San Jose.", "id": 15},
        {"text": "Warren Buffett's Berkshire Hathaway held its annual meeting in Omaha on Saturday.", "id": 16},
        {"text": "OpenAI released ChatGPT in November 2022, changing the AI landscape.", "id": 17},
        {"text": "Prime Minister Narendra Modi inaugurated the new parliament building in Delhi.", "id": 18},
        {"text": "Spotify founder Daniel Ek announced layoffs at their Stockholm office.", "id": 19},
        {"text": "The European Central Bank raised interest rates in Frankfurt last Thursday.", "id": 20},
        # Healthcare/Clinical examples
        {"text": "Patient John Smith, age 45, was prescribed Metformin 500mg twice daily.", "id": 21},
        {"text": "Dr. Emily Watson at Mayo Clinic diagnosed the condition on February 15.", "id": 22},
        {"text": "The FDA approved Pfizer's new vaccine for distribution in the United States.", "id": 23},
        {"text": "Memorial Hospital in Chicago reported increased cases of influenza this winter.", "id": 24},
        {"text": "Nurse practitioner Maria Garcia administered the Moderna booster shot.", "id": 25},
    ]

    output_path = DATASETS_DIR / "ner" / "sentences.json"
    with open(output_path, "w") as f:
        json.dump(sentences, f, indent=2)
    print(f"Created: {output_path} ({len(sentences)} samples)")


def create_image_classification_dataset():
    """
    For image classification, we'll create a manifest file pointing to sample images.
    Students can download images or use the provided ones.
    """
    (DATASETS_DIR / "image_classification").mkdir(exist_ok=True)
    # Using placeholder images from picsum.photos for demo
    # In real lab, students would use their own images
    images = [
        {"image": "https://picsum.photos/seed/cat1/400/300", "id": 1, "hint": "animal"},
        {"image": "https://picsum.photos/seed/dog1/400/300", "id": 2, "hint": "animal"},
        {"image": "https://picsum.photos/seed/car1/400/300", "id": 3, "hint": "vehicle"},
        {"image": "https://picsum.photos/seed/bike1/400/300", "id": 4, "hint": "vehicle"},
        {"image": "https://picsum.photos/seed/house1/400/300", "id": 5, "hint": "building"},
        {"image": "https://picsum.photos/seed/tree1/400/300", "id": 6, "hint": "nature"},
        {"image": "https://picsum.photos/seed/food1/400/300", "id": 7, "hint": "food"},
        {"image": "https://picsum.photos/seed/phone1/400/300", "id": 8, "hint": "electronics"},
        {"image": "https://picsum.photos/seed/book1/400/300", "id": 9, "hint": "object"},
        {"image": "https://picsum.photos/seed/flower1/400/300", "id": 10, "hint": "nature"},
        {"image": "https://picsum.photos/seed/laptop1/400/300", "id": 11, "hint": "electronics"},
        {"image": "https://picsum.photos/seed/beach1/400/300", "id": 12, "hint": "nature"},
        {"image": "https://picsum.photos/seed/city1/400/300", "id": 13, "hint": "urban"},
        {"image": "https://picsum.photos/seed/mountain1/400/300", "id": 14, "hint": "nature"},
        {"image": "https://picsum.photos/seed/coffee1/400/300", "id": 15, "hint": "food"},
    ]

    output_path = DATASETS_DIR / "image_classification" / "images.json"
    with open(output_path, "w") as f:
        json.dump(images, f, indent=2)
    print(f"Created: {output_path} ({len(images)} samples)")

    # Also create a README for this dataset
    readme = """# Image Classification Dataset

This dataset contains 15 sample images for classification practice.

## Categories to use:
- animal
- vehicle
- building
- nature
- food
- electronics
- urban
- object

## Label Studio Import:
1. Create new project with Image Classification template
2. Import `images.json`
3. Configure labels in the labeling interface

## Note:
Images are loaded from picsum.photos (placeholder service).
For production, use your own images.
"""
    readme_path = DATASETS_DIR / "image_classification" / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)


def create_object_detection_dataset():
    """
    For object detection, we need images with multiple objects.
    Using URLs to sample images that have clear objects.
    """
    (DATASETS_DIR / "object_detection").mkdir(exist_ok=True)
    # Sample images good for object detection practice
    images = [
        {
            "image": "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=640",
            "id": 1,
            "description": "Red sports car on road"
        },
        {
            "image": "https://images.unsplash.com/photo-1517649763962-0c623066013b?w=640",
            "id": 2,
            "description": "Cyclists racing"
        },
        {
            "image": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=640",
            "id": 3,
            "description": "Food plate with multiple items"
        },
        {
            "image": "https://images.unsplash.com/photo-1533743983669-94fa5c4338ec?w=640",
            "id": 4,
            "description": "Office desk with laptop, phone, coffee"
        },
        {
            "image": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=640",
            "id": 5,
            "description": "Classic car"
        },
        {
            "image": "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=640",
            "id": 6,
            "description": "Dog in park"
        },
        {
            "image": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640",
            "id": 7,
            "description": "Cat close-up"
        },
        {
            "image": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=640",
            "id": 8,
            "description": "Watch product photo"
        },
        {
            "image": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=640",
            "id": 9,
            "description": "Headphones product photo"
        },
        {
            "image": "https://images.unsplash.com/photo-1561948955-570b270e7c36?w=640",
            "id": 10,
            "description": "Cat sitting"
        },
    ]

    output_path = DATASETS_DIR / "object_detection" / "images.json"
    with open(output_path, "w") as f:
        json.dump(images, f, indent=2)
    print(f"Created: {output_path} ({len(images)} samples)")

    readme = """# Object Detection Dataset

This dataset contains 10 sample images for bounding box annotation practice.

## Suggested Labels:
- car
- person
- bicycle
- dog
- cat
- laptop
- phone
- cup/mug
- food
- watch
- headphones

## Label Studio Import:
1. Create new project with Object Detection template
2. Import `images.json`
3. Draw bounding boxes around objects
4. Assign labels to each box

## Tips:
- Draw boxes tightly around objects (small margin)
- Don't cut off parts of objects
- Label partially visible objects if >20% visible
"""
    readme_path = DATASETS_DIR / "object_detection" / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)


def create_label_studio_configs():
    """Create Label Studio XML config templates for each task."""

    configs = {
        "text_classification": """<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text" choice="single" showInLine="true">
    <Choice value="Positive" background="green"/>
    <Choice value="Negative" background="red"/>
    <Choice value="Neutral" background="gray"/>
  </Choices>
</View>""",

        "ner": """<View>
  <Labels name="label" toName="text">
    <Label value="PERSON" background="#FF0000"/>
    <Label value="ORG" background="#00FF00"/>
    <Label value="LOCATION" background="#0000FF"/>
    <Label value="DATE" background="#FFFF00"/>
    <Label value="PRODUCT" background="#FF00FF"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>""",

        "image_classification": """<View>
  <Image name="image" value="$image"/>
  <Choices name="category" toName="image" choice="single">
    <Choice value="animal"/>
    <Choice value="vehicle"/>
    <Choice value="building"/>
    <Choice value="nature"/>
    <Choice value="food"/>
    <Choice value="electronics"/>
    <Choice value="urban"/>
    <Choice value="object"/>
  </Choices>
</View>""",

        "object_detection": """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="car" background="#FF0000"/>
    <Label value="person" background="#00FF00"/>
    <Label value="dog" background="#0000FF"/>
    <Label value="cat" background="#FFFF00"/>
    <Label value="laptop" background="#FF00FF"/>
    <Label value="phone" background="#00FFFF"/>
    <Label value="cup" background="#FFA500"/>
    <Label value="food" background="#800080"/>
    <Label value="watch" background="#008080"/>
    <Label value="headphones" background="#FFC0CB"/>
  </RectangleLabels>
</View>"""
    }

    for task, config in configs.items():
        output_path = DATASETS_DIR / task / "label_studio_config.xml"
        with open(output_path, "w") as f:
            f.write(config)
        print(f"Created: {output_path}")


def create_main_readme():
    """Create main README for the datasets."""
    readme = """# Week 03 Lab: Data Annotation Datasets

Sample datasets for practicing data annotation with Label Studio.

## Quick Start

```bash
# Install Label Studio
pip install label-studio

# Start Label Studio
label-studio start

# Access at http://localhost:8080
```

## Datasets

| Task | Dataset | Samples | File |
|------|---------|---------|------|
| Text Classification | Movie Reviews | 30 | `text_classification/movie_reviews.json` |
| Named Entity Recognition | News Sentences | 25 | `ner/sentences.json` |
| Image Classification | Various Images | 15 | `image_classification/images.json` |
| Object Detection | Product/Animal Photos | 10 | `object_detection/images.json` |

## How to Import into Label Studio

1. **Create a new project** in Label Studio
2. **Go to Settings > Labeling Interface**
3. **Copy the config** from `<task>/label_studio_config.xml`
4. **Import data** from `<task>/*.json`

## Label Studio Configs

Each dataset folder contains a `label_studio_config.xml` file with the
recommended labeling interface configuration.

## Tasks

### 1. Text Classification
Classify movie reviews as Positive, Negative, or Neutral.
- Discuss edge cases (sarcasm, mixed sentiment)
- Calculate inter-annotator agreement with a partner

### 2. Named Entity Recognition
Tag entities in sentences: PERSON, ORG, LOCATION, DATE, PRODUCT.
- Pay attention to entity boundaries
- Handle nested entities appropriately

### 3. Image Classification
Categorize images into predefined categories.
- Note: Images loaded from URLs (requires internet)

### 4. Object Detection
Draw bounding boxes around objects in images.
- Practice tight box fitting
- Handle occlusion cases

## Calculating Agreement

After annotating, export your labels and calculate Cohen's Kappa:

```python
from sklearn.metrics import cohen_kappa_score

annotator1 = ['pos', 'neg', 'pos', 'neu', ...]
annotator2 = ['pos', 'neg', 'neu', 'neu', ...]

kappa = cohen_kappa_score(annotator1, annotator2)
print(f"Cohen's Kappa: {kappa:.2f}")
```
"""
    output_path = DATASETS_DIR / "README.md"
    with open(output_path, "w") as f:
        f.write(readme)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    print("Creating sample datasets for Label Studio...\n")

    create_text_classification_dataset()
    create_ner_dataset()
    create_image_classification_dataset()
    create_object_detection_dataset()
    create_label_studio_configs()
    create_main_readme()

    print("\nDone! Datasets created in:", DATASETS_DIR)
    print("\nTo use with Label Studio:")
    print("  1. pip install label-studio")
    print("  2. label-studio start")
    print("  3. Import the JSON files into your projects")
