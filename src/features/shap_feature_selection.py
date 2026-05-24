import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT/"data"/"processed"

dataset_path=PROCESSED_DATA_DIR/"GSPC_targeted.csv"
df=pd.read_csv(dataset_path)
print(f"columns: {df.columns}")
drop_columns=["date","future_5day_mean","target","ticker"]

X=df.drop(columns=drop_columns,errors="ignore")
y=df["target"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=False)

model=LGBMClassifier(n_estimators=200,learning_rate=0.05,max_depth=6,random_state=42)
model.fit(X_train,y_train)

explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(X_test)
shap.summary_plot(shap_values,X_test)

if isinstance(shap_values, list):
    shap_array = shap_values[1]

else:
    shap_array = shap_values
feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance": np.abs(shap_array).mean(axis=0)
})
feature_importance = (feature_importance .sort_values(by="importance",ascending=False))
print(feature_importance)

save_path = (
    PROCESSED_DATA_DIR/"feature_importance.csv"
)
feature_importance.to_csv(save_path,index=False)

print(f"saved: {save_path}")

TOP_N = 32

top_features = (
    feature_importance
    .head(TOP_N)
)

print(top_features)