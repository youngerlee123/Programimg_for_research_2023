from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from collections import Counter
import numpy as np
class Decisiontree:
    def __init__(self,max_depth=None) : #初始化資料 樹最大深度設置
        self.max_depth=max_depth

    def gini(self,classes): #用於計算每個節點的gini不純度
        total=len(classes) #計算樣本總數
        class_count={label: classes.count(label) for label in set(classes)} #計算各個類別出現次數(CLASS為測試資料中的類別項名稱)
        gini_impurity=1.0 #gini不純度=1-(所有類別各自出現機率*平方之後加總)
        for label in class_count:
            class_prob = class_count[label] / total
            gini_impurity -= class_prob**2
        return gini_impurity
    
    def best_feature(self, features , classes): #用於尋找最適合用於分類的特徵，這邊會將每個特徵的每個值都進行一次分裂計算gini 找出分類效果最好的特徵與值
        best_gini=1.0
        best_feature=None
        best_value=None

        for feature_idx in range(len(features[0])):  # 遍歷每個特徵
            feature_values = set([sample[feature_idx] for sample in features])  # 獲取該特徵的所有值
            for value in feature_values:  # 遍歷特徵的每個值，計算分裂後的 Gini 不純度
                left_split, right_split, left_classes, right_classes = self.split_data(features, classes, feature_idx, value)
                gini = (len(left_classes) / len(classes)) * self.gini(left_classes) + (len(right_classes) / len(classes)) * self.gini(right_classes)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_value = value

        if best_feature is None or best_value is None:
        # 如果找不到適合的分裂特徵或值，則回傳出現最多的標籤
            return max(set(classes), key=classes.count), None
        print("best_feature is",best_feature) #測試用 看他有沒有找對
        print("best_value is",best_value) #測試用 看他有沒有找對
        return best_feature,best_value 

    def split_data(self, features, classes, best_feature, best_value): #分裂 features是fit中給定的特徵集，classes則是類別集
        left_split = [] #左半分裂後的資料存放
        right_split = [] #右半分裂後的資料存放
        left_classes = [] #左半分裂後的類別存放
        right_classes = [] #右半分裂後的類別存放
            
    
        for i in range(len(features)): #分類左右子集，小於等於分裂閥值則為左子集，大於分裂閥值則為右子集
            if features[i][best_feature] <= best_value:  # 這裡比較指定特徵的值
                left_split.append(features[i])
                left_classes.append(classes[i])
            else:
                right_split.append(features[i])
                right_classes.append(classes[i])    

        return left_split, right_split, left_classes, right_classes


    def tree(self, features, classes , depth=0): #配合前面製作分類樹
        #print("目前子集",classes) #debug用
        if len(set(classes))==1: #如果只剩一個類別就不需要繼續分裂了，回傳該標籤 
            return classes[0]
        print ("now depth:", depth) #debug用
        if self.max_depth is not None and depth >= self.max_depth: #檢查是否達到了設定的最大深度。如果是，回傳該集最頻繁出現的類別。
            return max(set(classes), key=classes.count) #將出現最多的classes轉換為集合

        split_feature, split_value = self.best_feature(features, classes) #透過best feature來執行分裂
        left_split, right_split, left_classes, right_classes = self.split_data(features, classes, split_feature, split_value)

        if not left_split or not right_split: #檢查左右是否有一個子集空 有則回傳該集最頻繁出現的類別。
            return max(set(classes), key=classes.count) 

        left_subtree = self.tree(left_split, left_classes, depth + 1) #遞迴地建立左子集並計算深度
        right_subtree = self.tree(right_split, right_classes, depth + 1) #遞迴地建立右子集並計算深度

        return (split_feature, split_value, left_subtree, right_subtree)
    
    def fit(self, features, classes): #建立設計好的Decision Tree 模型
        self.decision_tree = self.tree(features, classes)

    # def predict_single(self, sample):
    #     node = self.decision_tree
    #     while isinstance(node, tuple):
    #         split_feature, split_value, left_subtree, right_subtree = node
    #         if sample[0] <= split_value:
    #             node = left_subtree
    #         else:
    #             node = right_subtree
    #     return node  
    def predict_single(self, sample):
        sample_list = list(sample)  # 將 sample 從元組轉換為 list
        node = self.decision_tree
        while isinstance(node, tuple):
            split_feature, split_value, left_subtree, right_subtree = node
            if sample_list[0] <= split_value:  # 使用 list 形式的特徵值
                node = left_subtree
            else:
                node = right_subtree
        return node

    def predict(self, samples):
        return [self.predict_single(sample) for sample in samples] #基於前面對單個樣本的預測進行全部整組的預測
    
    def gini_importance(self, features, classes):
            total_samples = len(classes)
            class_counts = Counter(classes)
            impurity_before_split = 1.0

            for label in class_counts:
                class_prob = class_counts[label] / total_samples
                impurity_before_split -= class_prob ** 2

            gini_importance_scores = []

            for feature_idx in range(len(features[0])):  # 修改此處
                feature_values = [sample[feature_idx] for sample in features]
                unique_values = set(feature_values)
                weighted_impurity_reduction = 0.0

                for value in unique_values:
                    left_indices = [i for i, val in enumerate(feature_values) if val <= value]
                    right_indices = [i for i, val in enumerate(feature_values) if val > value]

                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue

                    left_classes = [classes[i] for i in left_indices]
                    right_classes = [classes[i] for i in right_indices]

                    left_weight = len(left_indices) / total_samples
                    right_weight = len(right_indices) / total_samples

                    impurity_left = self.gini(left_classes)
                    impurity_right = self.gini(right_classes)
                    weighted_impurity_reduction += (impurity_before_split - (left_weight * impurity_left + right_weight * impurity_right))

                gini_importance_scores.append((feature_idx, weighted_impurity_reduction))

            gini_importance_scores.sort(key=lambda x: x[1], reverse=True)
            return gini_importance_scores

#---------------------------------------------------------------------------------------------------------------------實際測試用
import csv
def read_csv_file(file_path):
    features = []
    classes = []
    with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader) #跳過第一行標題
        for row in csvreader:
            # 假設 CSV 檔案的格式為 "特徵1,特徵2,特徵3,標籤"
            Class = row[-1]  # 最後一列是標籤
            feature_row = [float(value) for value in row[:-1]]  # 將特徵轉換為浮點數列表
            features.append(feature_row)
            classes.append(Class)
    return features, classes

# 主程式

# 讀取 CSV 檔案
file_path = "data0823.csv"  # 將 "your_csv_file.csv" 替換為實際的 CSV 檔案路徑
features, classes = read_csv_file(file_path)
# print(classes)
# print(features)
#建立並訓練 DecisionTree 模型
tree = Decisiontree(max_depth=5)
tree.fit(features[:30], classes[:30])


# 進行預測
predictions = tree.predict(features[30:])
print(predictions)

importances = tree.gini_importance(features[:30], classes[:30])
top_n = 20  
for i, (feature_idx, importance) in enumerate(importances[:top_n]):
    print(f"Rank {i+1}: Feature {feature_idx} importance: {importance}")



#---------------------------------------------------------------------------------------------------------------------package測試用
# tree = DecisionTreeClassifier(max_depth=1)
# tree.fit(features[:24], classes[:24])
# result = permutation_importance(tree, features, classes, n_repeats=30, random_state=0)
# feature_importances = result.importances_mean
# # 列印前 20 個特徵的重要性
# for i, importance in enumerate(feature_importances[:20]):
#     print(f"Feature {i}: Importance = {importance}")
#---------------------------------------------------------------------------------------------------------------------測試debug用
# features = [(22.0,), (33.0,), (54.0,), (48.0,), (65.0,), (74.0,), (26.0,), (17.0,), (62.0,), (56.0,), (42.0,), (6.0,), (15.0,), (38.0,), (71.0,), (59.0,)]  # 注意特徵值用元組形式 (2.0,) 而不是 [2.0]
# classes = ['20', '30', '50', '40', '60', '70', '20', '10', '60', '50', '40', '0', '10', '30', '70', '50']

# tree = Decisiontree(max_depth=2)
# tree.fit(features, classes)
# predictions = tree.predict([(54.0,), (42.0,), (22.0,), (51.0,), (13.0,)])  # 注意特徵值用元組形式 (2.5,) 而不是 [2.5]
# print(predictions)  # Output: ['A', 'B']
#-----------------------------------------------------------------------------
# features = [[2.0], [3.0], [4.0], [5.0], [6.0]]
# classes = ['A', 'A', 'B', 'B', 'B']

# tree = Decisiontree(max_depth=2)
# tree.fit(features, classes)
# predictions = tree.predict([[2.5], [3.5]])  # 注意特徵值用元組形式 (2.5,) 而不是 [2.5]
# print(predictions)  # Output: ['A', 'B']




