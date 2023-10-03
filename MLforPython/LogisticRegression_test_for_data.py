
#----------------------------------------------------------------------------------------------------------------------------------------package method
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_csv_file(file_path):
    features = []  # 特徵
    classes = []   # 類別
    feature_names = []  # 特徵名
    with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        feature_names = next(csvreader)
        next(csvreader)
        for row in csvreader:
            # 假設CSV 文件的格式為 "特徵1,特徵2,...,標籤"
            Class = int(row[-1])  # 最后一列是類別，轉為整數
            feature = [float(value) for value in row[:-1]]  # 前面的列是特徵，轉為浮點數
            features.append(feature)
            classes.append(Class)
    return features, classes, feature_names

file_path = "data0729.csv"  
features, classes, feature_names = read_csv_file(file_path)
#features 和 classes 轉換為array
X = np.array(features)  
y = np.array(classes)   



X_train = X[:42]  
y_train = y[:42]
X_test = X[42:51]  
y_test = y[42:51]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# 測試模型性能
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
# 獲得特徵重要分數
feature_weights = model.coef_[0]

# 列出特徵重要前十
top_10_feature_indices = np.argsort(feature_weights)[-10:]  #排名
top_10_feature_weights = feature_weights[top_10_feature_indices] #各自分數 

# 輸出前十名的特徵與其分數
print("Top 10 Features with Highest Weights:")
for i, feature_index in enumerate(top_10_feature_indices):
    weight = top_10_feature_weights[i]
    feature_name = feature_names[feature_index]
    print(f"Feature '{feature_name}': Weight = {weight}")

print("Accuracy:", accuracy)#輸出預測準確率
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print("Predicted Classes for 測試集:", y_pred)