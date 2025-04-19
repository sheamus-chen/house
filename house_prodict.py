import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 模擬資料集
data = {
    'size_sqft': [600, 800, 1000, 1200, 1500, 1800, 2000, 2200],
    'num_rooms': [2, 3, 3, 4, 4, 5, 5, 6],
    'age_years': [30, 20, 15, 10, 8, 5, 3, 1],
    'price': [100000, 150000, 180000, 220000, 250000, 280000, 310000, 350000]
}

# 建立 DataFrame
df = pd.DataFrame(data)

# 特徵 (X) 和目標值 (y)
X = df[['size_sqft', 'num_rooms', 'age_years']]
y = df['price']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
predictions = model.predict(X_test)

# 顯示預測與實際值
for real, pred in zip(y_test, predictions):
    print(f"實際價格: {real}, 預測價格: {round(pred)}")

# 顯示模型係數
print("\n模型係數:", model.coef_)
print("截距:", model.intercept_)

# 畫圖（預測 vs 實際）
plt.scatter(y_test, predictions)
plt.xlabel("實際價格")
plt.ylabel("預測價格")
plt.title("預測 vs 實際房價")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.grid(True)
plt.show()