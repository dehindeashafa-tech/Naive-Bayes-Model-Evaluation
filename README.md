# Naive-Bayes-Model-Evaluation
Naive Bayes Classification – NBA Player Longevity Prediction

## Project Overview
This project aims to predict whether a drafted NBA player will play for 5 or more years (`target_5yrs`) using an engineered dataset of their rookie-year statistics. We employ a Gaussian Naive Bayes classification model, focusing on identifying key statistical indicators for career longevity and translating these predictions into actionable scouting insights.

## Dataset Summary
The dataset `/content/extracted_nba_playersr_data.csv` contains various numerical statistics for NBA players, primarily from their rookie year. The target variable, `target_5yrs`, is a binary indicator (1 for 5+ years, 0 for less than 5 years). The features include metrics such as field goal percentage (`fg`), three-point percentage (`3p`), free throw percentage (`ft`), rebounds (`reb`), assists (`ast`), steals (`stl`), blocks (`blk`), turnovers (`tov`), total points (`total_points`), and an `efficiency` score.

## Modeling Approach: Gaussian Naive Bayes
We utilized a Gaussian Naive Bayes (GNB) classifier from Scikit-learn. This model is a probabilistic algorithm based on Bayes' theorem, assuming that features are normally distributed and conditionally independent given the class label. Its simplicity and computational efficiency make it a good baseline for initial classification tasks.

## Top Predictive Features
Gaussian Naive Bayes does not directly provide a feature importance ranking in the same way tree-based models do. However, the model uses all the provided numerical features (fg, 3p, ft, reb, ast, stl, blk, tov, total_points, efficiency) to calculate the likelihood of a player belonging to each class. The influence of each feature is determined by its mean and variance within each class.

## Performance Metrics
The model's performance was evaluated using a Confusion Matrix, Precision, and Recall on a held-out test set (30% of the data).

### Confusion Matrix
```
[[True Negatives (TN)  False Positives (FP)]
 [False Negatives (FN) True Positives (TP)]]
```

Interpretation for our model results:
*   **True Negatives (TN):** Players correctly predicted to *not* play 5+ years.
*   **False Positives (FP):** Players predicted to play 5+ years but *did not* (identifying a 'bust').
*   **False Negatives (FN):** Players predicted to *not* play 5+ years but *did* (missing talent).
*   **True Positives (TP):** Players correctly predicted to play 5+ years.

(See the confusion matrix plot generated in the notebook for specific counts)

### Precision and Recall
These metrics are particularly relevant for scouting:

*   **Precision (Minimize False Positives: 'busts'):** Of all players predicted to play 5+ years, how many actually did? High precision reduces wasted draft picks.
    *   **Our Model's Precision:** `0.8302`

*   **Recall (Minimize False Negatives: missed talent):** Of all players who *actually* played 5+ years, how many did the model correctly identify? High recall ensures less missed talent.
    *   **Our Model's Recall:** `0.5366`

(Refer to cell `a084b128` for the latest calculated precision and recall values.)

## Assumption Analysis and Business Interpretations

For a detailed discussion on the Gaussian Naive Bayes' "Independence Assumption" and its realism for basketball statistics, along with a summary of the model's reliability and limitations for a scouting department audience, please refer to the Markdown cells labeled `Explain the Naive Bayes "Independence Assumption"` and `Summarize Model Reliability and Limitations for a Scouting Department Audience` within the notebook.
