import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. การนำเข้าข้อมูล
data = pd.read_csv('student_data.csv')  # ตรวจสอบให้แน่ใจว่าคุณมีไฟล์นี้ในโฟลเดอร์เดียวกัน

# 2. การแปลงค่าผลลัพธ์
data['Final_Result'] = data['Final_Result'].map({'Passed': 1, 'Failed': 0})

# 3. การจัดการกับข้อมูลที่หายไป
data.fillna(data.mean(numeric_only=True), inplace=True)  # คำนวณเฉพาะค่าที่เป็นตัวเลข

# 4. การเตรียมข้อมูล
X = data.drop(columns=['Student_ID', 'Final_Result'])  # 'Final_Result' คือค่าที่เราต้องทำนาย
y = data['Final_Result']  # 'Final_Result' คือผลลัพธ์ที่ต้องการทำนาย เช่น 'Passed' หรือ 'Failed'

# 5. แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. การสร้างโมเดล Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. บันทึกโมเดล
with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("โมเดล Decision Tree ถูกบันทึกเรียบร้อยแล้ว")

# 8. ทำนายผลลัพธ์จากชุดข้อมูลทดสอบ
y_pred = model.predict(X_test)

# 9. แสดงผลลัพธ์
accuracy = accuracy_score(y_test, y_pred)
print(f'ความแม่นยำ: {accuracy:.2f}')
print("\nรายงานการจำแนกประเภท:")
print(classification_report(y_test, y_pred))

# 10. แสดง confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Failed', 'Passed'], yticklabels=['Failed', 'Passed'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# 11. แสดงต้นไม้การตัดสินใจ
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['Failed', 'Passed'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()