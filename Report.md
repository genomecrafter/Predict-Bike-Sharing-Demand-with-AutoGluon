# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NIKITA S RAJ KAPINI

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
During submissions from five different experiments — including:

- Initial Raw Submission (`initial`)
- Feature-Engineered Submission (`add_features`)
- Hyperparameter Optimization:
  - HPO - Initial Setting
  - HPO - Setting 1
  - HPO - Setting 2 

I observed that some models produced negative values in the predictions. Since Kaggle rejects submissions with negative `count` values, it was necessary to preprocess the results by replacing all negative predictions with `0`.

Additionally, the output from the predictor only contained the predicted `count` values. To submit to Kaggle, I had to merge these predictions with the original `datetime` column from the test dataset and save the result as a two-column CSV file (`datetime`, `count`).

### Changes Made Before Submission:

- Replaced negative predictions with `0`
- Combined predictions with `datetime` from the test dataset
- Exported the final result to `submission.csv` in the required format

### What was the top ranked model that performed?
The top-performing model was with the best Kaggle score compared to other models:

- Model: WeightedEnsemble_L3	
- Validation RMSE: 33.868672

This model was developed by training on data obtained using exploratory data analysis and feature engineering without the use of a hyperparameter optimization routine.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
- The `datetime` column was parsed to extract key temporal features such as `year`, `month`, `day`, and `hour`. After extraction, the original `datetime` field was removed.
- The `season` and `weather` features were cast to categorical types, as they represent discrete categories rather than continuous values.
- New feature `day_type` was created using `holiday` and `workingday` to distinguish between "weekday", "weekend", and "holiday".
- Highly correlated features `casual` and `registered` were excluded from training since they appear only in the training set and not in the test data, despite showing strong correlation with the target.
- Due to a very high correlation (0.98) between `temp` and `atemp`, the `atemp` feature was dropped to avoid multicollinearity.
- Exploratory visualizations were performed to better understand distributions and relationships across features before modeling.

### How much better did your model preform after adding additional features and why do you think that is?
Feature engineering led to a notable boost in model performance. Converting numeric categorical variables to actual categorical types helped the model better interpret those inputs. Dropping `casual`, `registered`, and `atemp` reduced data leakage and multicollinearity. Breaking down the `datetime` feature into components like year, month, day, and hour, along with introducing a `day_type` feature, enabled the model to better capture seasonal trends and usage patterns.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tuning helped improve model performance over the baseline submission. Multiple configurations were tested during the tuning phase. Although hyperparameter tuned models delivered competitive performances in comparison to the model with EDA and added features, the latter performed exceptionally better on the test dataset.

**Observations:**
- AutoGluon's tuning is limited by the predefined hyperparameter ranges, which may restrict its ability to explore better combinations.
- The `time_limit` and `presets` settings play a critical role in successful tuning. If the time is too short or the configuration too demanding, model training may not complete.
- High-memory presets like `high_quality` were not feasible within the given resource constraints. Lighter alternatives such as `optimized_for_deployment` were preferred to ensure successful model builds.
- Balancing exploration and exploitation is key when tuning hyperparameters, especially under resource and time constraints.

### If you were given more time with this dataset, where do you think you would spend more time?
Given additional time, I would focus on experimenting with deep learning models, particularly neural networks, to evaluate their suitability for this regression problem. While traditional models performed well, it's worth exploring whether deep architectures could capture more complex patterns in the data. This would help determine if such models are too complex for the task or if they offer meaningful performance improvements over simpler algorithms.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|prescribed_values|prescribed_values|"presets: 'high quality' (auto_stack=True)"|1.80448|
|add_features|prescribed_values|prescribed_values|"presets: 'high quality' (auto_stack=True)"|0.46797|
|hpo (top-hpo-model: hpo2)|Tree-Based Models: (GBM, XT, XGB & RF)|KNN|"presets: 'optimize_for_deployment"|0.51152|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](images/train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](images/test_score.png)

## Summary
This project incorporated the **AutoGluon AutoML framework for Tabular Data** to predict bike-sharing demand with high accuracy and minimal manual tuning.

- The **AutoGluon framework** was thoroughly studied and integrated to build automated regression models for tabular data. It enabled rapid prototyping of a strong baseline model with minimal code and effort.
- Both **stack-ensembled** and **individually configured** models were trained, leveraging AutoGluon’s ability to handle feature preprocessing, model selection, and ensembling automatically.
- A significant improvement in performance was observed by combining **extensive exploratory data analysis (EDA)** and **feature engineering** with AutoGluon's base training pipeline, even **without hyperparameter tuning**.
- Further performance boosts were attempted through **automatic hyperparameter optimization**, model ensembling, and neural architecture search using AutoGluon. While this led to better results than the raw baseline, it **did not outperform the model trained on EDA-enhanced data with default settings**.
- It was observed that **hyperparameter tuning in AutoGluon** can be a **time-consuming and complex process**, heavily influenced by:
  - The allotted training time
  - Chosen presets
  - The family of models being tuned
  - The defined search space for hyperparameters
- Additionally, due to **version incompatibilities**, it was not feasible to run AutoGluon locally. Issues such as mismatched versions of `scikit-learn` and `xgboost` were encountered.
  - As a workaround, the entire pipeline was executed on **Kaggle Notebooks**, where necessary dependencies could be better managed.
  - After multiple trials, deprecated modules were replaced, configurations adjusted, and finally, the model ran successfully—bringing a great sense of accomplishment.

In summary, AutoGluon proved to be a powerful tool for tabular prediction tasks, especially when combined with domain-specific feature engineering. While hyperparameter tuning added complexity, it offered valuable insights into model behavior and limits.
