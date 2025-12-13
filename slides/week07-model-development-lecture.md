---
marp: true
theme: default
paginate: true
style: @import "custom.css";
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Model Development & Training

**CS 203: Software Tools and Techniques for AI**
Prof. Nipun Batra, IIT Gandhinagar

---

# Today's Agenda

- Model selection strategies
- Training best practices
- Hyperparameter optimization
- AutoML with AutoGluon
- Model checkpointing
- Transfer learning
- **Fine-tuning LLMs** (New!)
- Training pipelines

---

# Model Development Lifecycle

It's not just `model.fit()`. It's a cycle.

<div class="mermaid">
graph LR
    A[Data Prep] --> B[Feat Eng];
    B --> C[Model Selection];
    C --> D[Training];
    D --> E[Evaluation];
    E --> F{Good Enough?};
    F -- No --> C;
    F -- Yes --> G[Deployment];
    E --> H[Error Analysis];
    H --> A;
</div>

**Iterative Process**:
1. Start simple (Baseline).
2. Analyze errors.
3. Add complexity (New features, complex models).

---

# Model Selection Strategy

**Don't start with a Transformer.** Start with a baseline.

| Data Type | Baseline (Fast, Simple) | Advanced (SOTA, Heavy) |
| :--- | :--- | :--- |
| **Tabular** | Logistic Regression, Decision Tree | XGBoost, LightGBM, TabNet |
| **Image** | ResNet-18 | EfficientNet, ViT (Vision Transformer) |
| **Text** | TF-IDF + Naive Bayes | BERT, RoBERTa, GPT (Fine-tuned) |
| **Time Series** | ARIMA, Linear Regression | LSTM, Transformer |

**Why Baselines?**
- Debug pipeline bugs quickly.
- Establish a "floor" for performance.
- If deep learning only gives +1% over Logistic Regression, is it worth the cost?

---

# Model Selection Criteria

**Consider when choosing models**:

1. **Data size**: Small data → simpler models (avoid overfitting)
2. **Feature types**: Mixed types → tree-based models
3. **Interpretability**: Need explanations → linear models, decision trees
4. **Latency requirements**: Real-time → fast models (linear, small trees)
5. **Compute budget**: Limited resources → avoid deep learning
6. **Maintenance**: Production → stable, well-supported models

**No Free Lunch Theorem**: No single best model for all problems.

---

# Evaluation Metrics

**Classification metrics**:

| Metric | Formula | Use When |
| :--- | :--- | :--- |
| **Accuracy** | (TP + TN) / Total | Balanced classes |
| **Precision** | TP / (TP + FP) | Minimize false positives |
| **Recall** | TP / (TP + FN) | Minimize false negatives |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance P and R |
| **AUC-ROC** | Area under ROC curve | Overall model quality |

**Regression metrics**:
- **MAE** (Mean Absolute Error): Average error magnitude
- **MSE** (Mean Squared Error): Penalizes large errors
- **RMSE** (Root MSE): Same units as target
- **R²** (R-squared): Proportion of variance explained

---

# Confusion Matrix Deep Dive

**For binary classification**:

```
                Predicted
              Positive  Negative
Actual Pos      TP        FN
       Neg      FP        TN
```

**Example**: Cancer detection
- **TP** (True Positive): Correctly identified cancer
- **FP** (False Positive): Healthy flagged as cancer (unnecessary treatment)
- **FN** (False Negative): Cancer missed (dangerous!)
- **TN** (True Negative): Correctly identified healthy

**Choosing metric**:
- Medical diagnosis → **Maximize Recall** (catch all positives)
- Spam filter → **Maximize Precision** (don't block real emails)

---

# Precision vs Recall Trade-off

**Setting decision threshold**:

```python
# Default threshold = 0.5
y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)

# High precision (few false positives)
y_pred_high_precision = (model.predict_proba(X)[:, 1] >= 0.8).astype(int)

# High recall (few false negatives)
y_pred_high_recall = (model.predict_proba(X)[:, 1] >= 0.2).astype(int)
```

**Precision-Recall curve**: Visualize trade-off at all thresholds.

**ROC curve**: True Positive Rate vs False Positive Rate.

**AUC-ROC = 1.0**: Perfect classifier
**AUC-ROC = 0.5**: Random guessing

---

# Bias vs. Variance Trade-off

**Bias (Underfitting)**: Model is too simple to capture patterns.
**Variance (Overfitting)**: Model memorizes noise in training data.

<div class="mermaid">
graph TD
    A[Total Error] --> B[Bias^2];
    A --> C[Variance];
    A --> D[Irreducible Error];
</div>

**Goal**: Sweet spot where Total Error is minimized.

**Fixing Overfitting**:
- More data
- Regularization (L1/L2, Dropout)
- Simpler model

**Fixing Underfitting**:
- More features
- Complex model
- Train longer

---

# Learning Curves: Diagnosing Problems

**Training curve** (loss over epochs):
- Still decreasing → train more
- Flattened → training complete

**Learning curve** (performance vs data size):
- High bias (underfitting): Both curves plateau at low performance
- High variance (overfitting): Large gap between train/val

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
```

**Interpretation**:
- Gap closing with more data → get more data
- Gap persistent → simpler model or regularization

---

# Regularization Techniques

**Prevent overfitting by adding constraints**.

**L1 Regularization (Lasso)**:
- Loss = MSE + λ × Σ|w|
- Drives some weights to exactly zero
- **Effect**: Feature selection (sparse models)

**L2 Regularization (Ridge)**:
- Loss = MSE + λ × Σw²
- Shrinks all weights toward zero
- **Effect**: Reduces model complexity

**ElasticNet**: Combines L1 + L2

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# L2 regularization
model = Ridge(alpha=1.0)  # alpha = λ

# L1 regularization
model = Lasso(alpha=1.0)

# Both
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

---

# Dropout (Neural Networks)

**Randomly drop neurons during training**.

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        x = self.fc2(x)
        return x
```

**Why it works**:
- Forces network to not rely on specific neurons
- Acts like training ensemble of sub-networks
- Disabled during inference

---

# Data Splitting Strategies

**1. Hold-out Set**:
- Train (60%), Validation (20%), Test (20%).
- **Risk**: Validation set might be lucky/unlucky.

**2. K-Fold Cross-Validation**:
- Robust estimate of performance.
- Train K times on K different splits.

**3. Stratified K-Fold**:
- Maintains class distribution in each fold
- **Critical for imbalanced datasets**

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train model...
```

---

# Time Series Splitting

**Problem**: Can't shuffle time series data (leakage!).

**Time Series CV**:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # Always: train comes before test
```

**Expanding window**: Train set grows each fold
**Sliding window**: Fixed-size train set

**Walk-forward validation**: Retrain after each prediction.

---

# Feature Engineering Principles

**Good features** = better performance than complex models.

**Types of features**:

1. **Numerical transformations**:
   - Log transform (reduce skew)
   - Polynomial features (capture non-linearity)
   - Binning/discretization

2. **Categorical encoding**:
   - One-hot encoding (low cardinality)
   - Target encoding (high cardinality)
   - Frequency encoding

3. **Date/time features**:
   - Extract: hour, day, month, day_of_week
   - Cyclical encoding: sin/cos for hour

4. **Domain-specific**:
   - Text: TF-IDF, n-grams
   - Images: SIFT, HOG (or CNN features)

---

# Feature Importance

**Identify which features matter**:

**Tree-based** (built-in):
```python
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values('importance', ascending=False)
```

**Permutation importance** (model-agnostic):
```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,
    random_state=42
)
```

**SHAP values** (best but slow):
- Explains individual predictions
- Shows feature contributions

---

# Hyperparameter Optimization (HPO)

**Parameters**: Learned from data (Weights, Biases).
**Hyperparameters**: Set *before* training (Learning Rate, Batch Size, Depth).

**Search Strategies**:

1.  **Grid Search**: Try *every* combination.
    - Safe but exponentially expensive ($O(N^D)$).
2.  **Random Search**: Randomly sample configurations.
    - Surprisingly effective.
3.  **Bayesian Optimization (Optuna)**: Smart search.
    - "Given that `lr=0.1` was bad, don't try `lr=0.2`, try `lr=0.01`."

---

# Bayesian Optimization Visualized

<div class="columns">

<div>

**How it works**:
1.  Build a probability model of the objective function.
2.  Choose next hyperparameter to query (Exploration vs Exploitation).
3.  Update model.

</div>

<div>

**Optuna Code**:
```python
def objective(trial):
    # Suggest params
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    depth = trial.suggest_int('depth', 3, 10)
    
    model = Train(lr, depth)
    return model.val_accuracy

study.optimize(objective, n_trials=100)
```

</div>

</div>

---

# Ensemble Methods

**Wisdom of crowds**: Combine multiple models for better performance.

**Three main strategies**:

1. **Bagging** (Bootstrap Aggregating)
   - Train models on random subsets of data
   - Average predictions (regression) or vote (classification)
   - Example: **Random Forest** (ensemble of decision trees)

2. **Boosting**
   - Train models sequentially
   - Each model corrects previous model's errors
   - Example: **XGBoost, LightGBM, AdaBoost**

3. **Stacking**
   - Train diverse base models
   - Meta-model learns to combine their predictions
   - Example: **AutoGluon**

---

# Bagging vs Boosting

| Aspect | Bagging | Boosting |
| :--- | :--- | :--- |
| **Training** | Parallel | Sequential |
| **Goal** | Reduce variance | Reduce bias |
| **Weighting** | Equal | Focuses on hard examples |
| **Overfitting** | Less prone | Can overfit |
| **Example** | Random Forest | XGBoost, AdaBoost |

**When to use**:
- **Bagging**: High variance models (deep trees)
- **Boosting**: High bias models (weak learners)

---

# Gradient Boosting Explained

**Intuition**: Each tree corrects residual errors of previous trees.

```python
# Simplified gradient boosting
predictions = 0

for i in range(n_trees):
    # Calculate residuals (errors)
    residuals = y_true - predictions

    # Train tree to predict residuals
    tree = DecisionTree().fit(X, residuals)

    # Update predictions
    predictions += learning_rate * tree.predict(X)
```

**Hyperparameters**:
- `n_estimators`: Number of trees (more = better, but slower)
- `learning_rate`: How much each tree contributes
- `max_depth`: Tree complexity (prevent overfitting)

---

# Class Imbalance Problem

**Scenario**: 99% normal transactions, 1% fraud.

**Naive accuracy**: Predict all "normal" → 99% accuracy (useless!).

**Solutions**:

1. **Resampling**:
   - **Oversample minority**: Duplicate rare class (SMOTE)
   - **Undersample majority**: Remove common class

2. **Class weights**:
   - Penalize misclassifying minority class more

3. **Ensemble methods**:
   - BalancedRandomForest, EasyEnsemble

4. **Evaluation metrics**:
   - Use F1, Precision-Recall, not Accuracy

---

# Handling Imbalanced Data

**SMOTE (Synthetic Minority Oversampling)**:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Class weights** (penalize minority class errors more):
```python
from sklearn.linear_model import LogisticRegression

# Automatically balance
model = LogisticRegression(class_weight='balanced')

# Manual weights
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

**Stratified sampling**: Ensure minority class in all splits.

---

# AutoML: Automated Machine Learning

**Philosophy**: "I don't care which model, just give me the best one."

**AutoGluon** (Amazon) creates a stacked ensemble of models.

<div class="mermaid">
graph TD
    Data --> M1[Random Forest];
    Data --> M2[CatBoost];
    Data --> M3[Neural Net];
    M1 --> L2[Weighted Ensemble];
    M2 --> L2;
    M3 --> L2;
    L2 --> Prediction;
</div>

**Pros**: SOTA performance with 3 lines of code.
**Cons**: Slow training, heavy inference, hard to interpret.

---

# AutoML Code Example

```python
from autogluon.tabular import TabularPredictor

# That's it!
predictor = TabularPredictor(label='target').fit(train_data)

# Evaluate
predictor.leaderboard(test_data)

# Predict
predictions = predictor.predict(test_data)
```

**What AutoGluon does**:
1. Feature engineering (one-hot encoding, etc.)
2. Trains 10-20 different models
3. Stacks them into ensemble
4. Hyperparameter tuning

**Use when**: Time-constrained, need best performance, don't need interpretability.

---

# Training Pipelines

Spaghetti code in notebooks is the enemy of reproducibility.
**Pipeline**: A reproducible recipe.

`Raw Data` -> `Imputer` -> `Scaler` -> `Encoder` -> `Model`

**scikit-learn Pipeline**:
```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('clf', RandomForestClassifier())
])
pipe.fit(X_train, y_train)
```
**Benefits**: No data leakage! Transforms are fit *only* on train splits during CV.

---

# Checkpointing & Early Stopping

**The "Epoch" Dilemma**:
- Train too little -> Underfit.
- Train too much -> Overfit.

**Early Stopping**:
- Monitor Validation Loss.
- If it stops decreasing for `patience` epochs -> **STOP**.

**Checkpointing**:
- Save the model weights *every time* validation loss improves.
- Restore the *best* version at the end.

---

# Transfer Learning: Theory

**Don't reinvent the wheel.**
Someone (Google/Meta) spent $10M to train a model on ImageNet (14M images). It learned to see edges, textures, shapes.

**Strategies**:
1.  **Feature Extraction**: Freeze backbone, train only the head (classifier).
    - Fast, low data requirement.
2.  **Fine-Tuning**: Unfreeze backbone (or parts of it) and train with low learning rate.
    - Slower, needs more data, higher accuracy.

<div class="mermaid">
graph TD
    A[Pre-trained Backbone] --> B[New Head];
    subgraph "Feature Extraction"
    A:::frozen
    B:::trainable
    end
    classDef frozen fill:#eee,stroke:#333,stroke-dasharray: 5 5;
    classDef trainable fill:#bbf,stroke:#333;
</div>

---

# Fine-Tuning LLMs (PEFT & LoRA)

**The Problem**: Fine-tuning a 7B parameter model requires ~100GB+ VRAM.
**Solution**: Parameter-Efficient Fine-Tuning (PEFT).

**LoRA (Low-Rank Adaptation)**:
- Freeze original weights $W$.
- Add small trainable rank decomposition matrices $A$ and $B$.
- $W' = W + BA$
- Trainable parameters reduced by 10,000x!

**Use Cases**:
- Adapting generic LLM to specific domain (Medical, Legal).
- Changing style/tone (Chatbot persona).

---

# Hugging Face PEFT Example

```python
from peft import LoraConfig, get_peft_model, TaskType

# 1. Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=8,            # Rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1
)

# 2. Wrap Base Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = get_peft_model(model, peft_config)

# 3. Train as usual (only 0.5% params are trainable!)
model.print_trainable_parameters()
```

---

# Experiment Tracking

**Problem**: "I trained 50 models. Which one had `lr=0.001` and `dropout=0.5`?"

**Solution**: Use a tracker (Weights & Biases, MLflow).

**What to track**:
- **Config**: Hyperparameters (yaml/json).
- **Metrics**: Loss, Accuracy, F1 (charts).
- **Artifacts**: Model weights (`.pt`), dataset versions.
- **System**: GPU usage, memory.

---

# Best Practices Summary

1.  **Baseline First**: Always beat a dummy classifier.
2.  **Leakage Free**: Use Pipelines.
3.  **Track Everything**: Use W&B/MLflow.
4.  **Save Often**: Checkpoints are life-savers.
5.  **Be Lazy**: Use Transfer Learning and AutoML where possible.

---

# Lab Preview

**Hands-on exercises:**
1.  **Manual**: Compare SVM vs Random Forest with Cross-Validation.
2.  **Automated**: Use AutoGluon to beat your manual models.
3.  **Optimization**: Use Optuna to tune the Random Forest.
4.  **Tracking**: Log everything to W&B.
5.  **(Advanced)**: Fine-tune a small BERT model using LoRA.

Let's code!
