## Business Understanding
Our goal is to:
- Come up with the best classification model to correctly classify the tickets based solely on the description.
- Provide the business with the best model and hyperparameters to drive the best ROC AUC score.
  - We choose ROC AUC score because:
    - We are doing classification and it is particulary subject to the [Accuracy Paradox](https://en.wikipedia.org/wiki/Accuracy_paradox) because of class imbalance in this set.
    - It is generally preferred for imbalanced data. In addition we should make sure to use OVR during grid search for this (imbalance) reason.
- Provide the business with alternative models trading ROC AUC score for performance in case that may be a concern or to scale for a much larger dataset that may need re-training.
