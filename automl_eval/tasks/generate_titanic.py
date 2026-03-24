"""
Script to generate a demo Titanic CSV dataset.
Run: python automl_eval/tasks/generate_titanic.py
"""
import csv, os, random

random.seed(42)
N = 300
OUTPUT = os.path.join(os.path.dirname(__file__), "titanic.csv")
COLUMNS = [
    "PassengerId","Survived","Pclass","Name","Sex",
    "Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked",
]

def rand_age():
    if random.random() < 0.2: return ""
    return f"{random.uniform(1,80):.1f}"

def rand_cabin():
    if random.random() < 0.7: return ""
    return f"{chr(65+random.randint(0,4))}{random.randint(1,99)}"

def rand_embarked():
    if random.random() < 0.02: return ""
    return random.choices(["S","C","Q"], weights=[0.7,0.2,0.1])[0]

rows = []
for i in range(1, N+1):
    rows.append([
        i,
        random.choices([0,1], weights=[0.6,0.4])[0],
        random.choices([1,2,3], weights=[0.25,0.25,0.50])[0],
        f"Person_{i}",
        random.choice(["male","female"]),
        rand_age(),
        random.choices(range(5), weights=[0.6,0.2,0.1,0.05,0.05])[0],
        random.choices(range(4), weights=[0.7,0.15,0.1,0.05])[0],
        f"T{random.randint(10000,99999)}",
        f"{random.expovariate(1/30):.2f}",
        rand_cabin(),
        rand_embarked(),
    ])

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
print(f"Written {len(rows)} rows to {OUTPUT}")
