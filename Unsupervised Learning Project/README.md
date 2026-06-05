# Customer Segmentation Using K-Means Clustering on Wine Quality Data

An unsupervised machine learning project that identifies natural groupings among red wine samples based on their chemical properties, using K-Means clustering and PCA visualization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Key Results](#key-results)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)
- [Future Work](#future-work)

---

## Project Overview

The goal of this project is to discover natural groupings among 1,599 red wine samples based on their chemical features — without using any labels — using the K-Means clustering algorithm. These clusters can help wine producers understand product segmentation and quality differentiation.

This is a classic **unsupervised learning** use case: no target variable is used during training. The wine quality score is only examined after clustering to evaluate how well the algorithm captured meaningful patterns.

---

## Dataset

**Wine Quality (Red)** — 1,599 samples, 11 chemical features

| Feature | Description |
|---|---|
| Fixed Acidity | Tartaric acid concentration |
| Volatile Acidity | Acetic acid (too high = vinegar taste) |
| Citric Acid | Adds freshness and flavour |
| Residual Sugar | Sugar remaining after fermentation |
| Chlorides | Salt content |
| Free Sulfur Dioxide | Prevents microbial growth |
| Total Sulfur Dioxide | Free + bound SO₂ |
| Density | Depends on alcohol and sugar content |
| pH | Acidity level (0–14 scale) |
| Sulphates | Wine additive, linked to SO₂ |
| Alcohol | Percentage alcohol by volume |

The `quality` column (score 0–10) is **excluded during training** and only used for post-hoc cluster validation.

---

## Project Workflow

1. **Data Loading** — Load the CSV dataset and inspect its shape and structure.
2. **Data Cleaning** — Check for missing values, remove duplicates, and filter outliers using Z-scores (threshold = 3).
3. **Exploratory Analysis** — Generate a correlation heatmap to understand feature relationships.
4. **Preprocessing** — Scale all features using `StandardScaler` to ensure equal contribution to distance calculations.
5. **Parameter Tuning (Elbow Method)** — Test k = 1 to 10 and plot SSE to identify the optimal number of clusters.
6. **Cluster Validation (Silhouette Score)** — Compute silhouette scores for k = 2 to 10 to confirm the cluster quality.
7. **K-Means Clustering** — Fit the final model with k = 3 and assign cluster labels.
8. **PCA Visualization** — Reduce to 2 dimensions and plot clusters to assess separation visually.
9. **Cluster Interpretation** — Compare average feature values and wine quality scores per cluster.

---

## Technologies Used

- **Python 3**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualizations
- **Scikit-learn** — `KMeans`, `PCA`, `StandardScaler`, `silhouette_score`
- **SciPy** — Z-score based outlier removal
- **Google Colab** — notebook environment

---
## Visualizations

### Correlation Heatmap
![Correlation Heatmap] 

### Elbow Method
![Elbow Method] 

### PCA Cluster Plot
![PCA Cluster Plot]!

### Cluster Feature Comparison
![Cluster Feature Comparison] 

## Key Results

**Optimal number of clusters: k = 3**, selected based on the Elbow Method (SSE stabilises after k = 3) and confirmed by Silhouette Scores.

| Cluster | Key Characteristics |
|---|---|
| Cluster 0 | Lower alcohol, moderate acidity |
| Cluster 1 | Higher alcohol, lower density |
| Cluster 2 | Moderate alcohol, higher density, lower pH |

Clusters with higher alcohol and balanced acidity tended to correspond to higher average wine quality, even though quality was never used during training — confirming that the clustering captured real chemical structure in the data.

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Upload `WineQT.csv` when prompted by the file upload cell.
3. Run all cells in order (`Runtime → Run all`).

### Option 2 — Local Jupyter

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
3. Place `WineQT.csv` in the same directory as the notebook.
4. Remove or comment out the Google Colab upload block and replace it with:
   ```python
   data = pd.read_csv("WineQT.csv")
   ```
5. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Unsupervised_learning.ipynb
   ```

---

## File Structure

```
├── Unsupervised_learning.ipynb   # Main notebook
├── WineQT.csv                    # Dataset (download separately)
└── README.md                     # This file
```

> The dataset can be downloaded from [Kaggle — Wine Quality](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).

---

## Future Work

- Try alternative clustering algorithms such as **DBSCAN** or **Hierarchical Clustering** and compare results.
- Experiment with **feature selection** to remove redundant correlated features before clustering.
- Tune outlier removal thresholds and evaluate the impact on cluster quality.
- Apply the same pipeline to the **white wine** dataset and compare segmentation patterns.