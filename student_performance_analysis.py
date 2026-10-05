import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student_performance_raw.csv")
df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
df["Category"] = df["Category"].astype(str).str.strip().str.upper().replace({"GENERAL":"General"})
for col in ["Attendance_Percent","Study_Hours_Per_Day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.drop_duplicates()
df["Attendance_Percent"] = df["Attendance_Percent"].fillna(df["Attendance_Percent"].median())
df["Study_Hours_Per_Day"] = df["Study_Hours_Per_Day"].fillna(df["Study_Hours_Per_Day"].median())

subjects=["Mathematics_Marks","Python_Marks","DSA_Marks","English_Marks"]
df["Total_Marks"]=df[subjects].sum(axis=1)
df["Average_Marks"]=df[subjects].mean(axis=1)

print(df[["Attendance_Percent","Study_Hours_Per_Day","Average_Marks"]].describe())
print("\nSubject averages:\n",df[subjects].mean())
print("\nGender averages:\n",df.groupby("Gender")["Average_Marks"].mean())
print("\nCategory averages:\n",df.groupby("Category")["Average_Marks"].mean())
print("\nAttendance correlation:",df["Attendance_Percent"].corr(df["Average_Marks"]))
print("Study-hours correlation:",df["Study_Hours_Per_Day"].corr(df["Average_Marks"]))

sns.set_theme(style="whitegrid")
plt.hist(df["Average_Marks"],bins=12,edgecolor="black"); plt.title("Distribution of Average Marks"); plt.show()
df[subjects].mean().plot(kind="bar"); plt.title("Average Marks by Subject"); plt.show()
plt.scatter(df["Attendance_Percent"],df["Average_Marks"]); plt.title("Attendance vs Average Marks"); plt.show()
sns.boxplot(data=df[subjects]); plt.title("Subject-wise Marks Distribution"); plt.show()
sns.heatmap(df[["Attendance_Percent","Study_Hours_Per_Day"]+subjects+["Average_Marks"]].corr(),annot=True,fmt=".2f"); plt.title("Correlation Heatmap"); plt.show()
sns.violinplot(x="Gender",y="Average_Marks",data=df); plt.title("Average Marks Distribution by Gender"); plt.show()
