"""
Test the full chain: automl_eval server <-> agentenv-automl <-> AgentGym client.

Verifies the connection works end-to-end using hardcoded actions
(no LLM needed). Run with all 3 servers up:

  Terminal 1: python run_server.py --tasks-dir automl_eval/tasks       (port 8766)
  Terminal 2: automl-eval-env --port 8080                               (port 8080)
  Terminal 3: python tests/test_agentgym_chain.py                       (this script)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

AGENTGYM_SERVER = "http://localhost:8080"


def main() -> None:
    # 1) Create environment instance (as AgentGym would)
    print("1) POST /create ...")
    resp = requests.post(f"{AGENTGYM_SERVER}/create", json={}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    env_id = data["id"]
    print(f"   env_id={env_id}")
    print(f"   observation (first 200 chars): {data['observation'][:200]}...")

    # 2) Reset to task 0 (titanic_binary)
    print("\n2) POST /reset ...")
    resp = requests.post(
        f"{AGENTGYM_SERVER}/reset",
        json={"id": env_id, "data_idx": 0},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"   observation (first 200 chars): {data['observation'][:200]}...")

    # 3) Step: send a PLAN
    plan = (
        "ACTION: PLAN\n"
        "1. Handle missing values with median imputation.\n"
        "2. Encode categorical features (Sex, Embarked) with LabelEncoder.\n"
        "3. Feature engineering: create FamilySize.\n"
        "4. Train a RandomForest model.\n"
        "5. Evaluate with ROC AUC.\n"
    )
    print("\n3) POST /step (PLAN) ...")
    resp = requests.post(
        f"{AGENTGYM_SERVER}/step",
        json={"id": env_id, "action": plan},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"   reward={data['reward']}, done={data['done']}")
    obs = data["observation"]
    # Show validator feedback
    if "Validator feedback" in obs:
        print("   " + obs[obs.index("--- Validator feedback ---"):])

    # 4) Step: send CODE
    code = '''ACTION: CODE
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

df = train_df.copy()
vdf = valid_df.copy()

df['Age'] = df['Age'].fillna(df['Age'].median())
vdf['Age'] = vdf['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna('S')
vdf['Embarked'] = vdf['Embarked'].fillna('S')

for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    vdf[col] = le.transform(vdf[col].astype(str))

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
vdf['FamilySize'] = vdf['SibSp'] + vdf['Parch'] + 1

feature_cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','FamilySize']
X_train = df[feature_cols]
y_train = df['Survived']
X_valid = vdf[feature_cols]
y_valid = vdf['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_pred_proba)
print(f"Validation ROC AUC: {auc:.4f}")
best_metric = auc
predictions = y_pred_proba
```'''
    print("\n4) POST /step (CODE) ...")
    resp = requests.post(
        f"{AGENTGYM_SERVER}/step",
        json={"id": env_id, "action": code},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"   reward={data['reward']}, done={data['done']}")
    obs = data["observation"]
    if "Validator feedback" in obs:
        print("   " + obs[obs.index("--- Validator feedback ---"):])

    # 5) Step: FINAL_SUBMIT
    print("\n5) POST /step (FINAL_SUBMIT) ...")
    resp = requests.post(
        f"{AGENTGYM_SERVER}/step",
        json={"id": env_id, "action": "ACTION: FINAL_SUBMIT"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"   reward={data['reward']}, done={data['done']}")
    obs = data["observation"]
    if "Episode finished" in obs:
        print("   " + obs[obs.index("=== Episode finished"):])

    # 6) Close
    print("\n6) POST /close ...")
    resp = requests.post(
        f"{AGENTGYM_SERVER}/close",
        json={"id": env_id},
        timeout=10,
    )
    print(f"   {resp.json()}")

    print("\n=== Full chain test PASSED ===")


if __name__ == "__main__":
    main()
