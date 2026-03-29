# Model Tuning TODO
## Goal: Tune XGBoost baseline for lower MAE via hyperparameter optimization.

1. ✅ [DONE] Create TODO.md
2. ✅ [DONE] Create notebooks/03-model-tuning.ipynb with tuning code
3. Install new deps if needed (Optuna): pip install optuna scikit-optimize
4. Run tuning notebook (GridSearchCV/RandomizedSearchCV or Optuna on params: n_estimators[100-500], max_depth[3-8], learning_rate[0.01-0.2], subsample[0.8-1], colsample_bytree[0.8-1])
5. Compare tuned MAE/R² vs baseline on test set (aim >5% MAE improvement)
6. Save best model as notebooks/tuned_salary_model.joblib
7. Update README.md and Progress.docx with results
8. attempt_completion

