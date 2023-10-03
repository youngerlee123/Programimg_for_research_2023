import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import os
import shap
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
data = pd.read_csv('testML0518.csv')

# 提取特徵和目標變數
X = data.iloc[:50, 2:2036]
y = data.iloc[:50, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)

rf = RandomForestClassifier(n_estimators=150, criterion='gini', 
                             max_depth=10, min_samples_split=2, 
                             min_samples_leaf=1, min_weight_fraction_leaf=0,
                             max_leaf_nodes=None, bootstrap=True, 
                             oob_score=True, n_jobs=1, random_state=None, 
                             verbose=0, warm_start=False, class_weight=None)

# 訓練模型
rf.fit(X_train, y_train)

# 選擇要可視化的樹的索引
tree_index = 0

# 導出樹的DOT格式
tree_dot = export_graphviz(rf.estimators_[tree_index], feature_names=X.columns, class_names=["0","1"], filled=True, rounded=True)

# 將DOT格式轉換為圖形
graph = graphviz.Source(tree_dot)

# 顯示圖形
graph.view()


y_pred = rf.predict(X_test)
# 顯示預測結果
for i in range(len(y_pred)):
    print(f"Sample {i+1}: Predicted = {y_pred[i]}, Actual = {y_test.iloc[i]}")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# 獲取特徵重要性
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# 繪製前二十名特徵的重要性長條圖
top_features = data.columns[2:2036][sorted_idx][:10]
top_feature_importance = feature_importance[sorted_idx][:10]

plt.barh(top_features, top_feature_importance)
plt.gca().invert_yaxis()  # 反轉 y 軸，讓最重要的特徵在上方
plt.title("Top 20 Feature Importance")
plt.xlabel("Feature Importance")
plt.show()


# # 使用 shap 解釋模型
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_test_predict)

# # 繪製 SHAP summary plot，顯示每個特徵的影響
# shap.summary_plot(shap_values[1], X_test_predict, feature_names=data.columns[2:2036])

# plt.show()


