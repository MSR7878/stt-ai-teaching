# Week 03 Lab: Data Annotation Datasets

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
