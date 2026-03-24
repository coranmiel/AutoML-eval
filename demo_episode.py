"""
Demo: one full episode in the AutoML RL environment.

Shows the cycle: reset -> observe -> step(PLAN) -> step(CODE) -> step(FINAL_SUBMIT).
Simulates agent behaviour with hardcoded actions.

Run:
    pip install pandas numpy scikit-learn
    python demo_episode.py
"""
import logging
from automl_eval.environment import AutoMLEnvironment
from automl_eval.task import Task
from automl_eval.task_registry import TaskRegistry
from automl_eval.reward import RewardWeights

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def build_demo_registry():
    registry = TaskRegistry()
    task = Task.from_json("automl_eval/tasks/titanic_binary.json")
    registry.register(task)
    return registry


AGENT_PLAN = """ACTION: PLAN
My plan for this binary classification task:
1. Handle missing values: impute Age with median, fill Embarked with mode, drop Cabin.
2. Encode categorical features: LabelEncode Sex and Embarked.
3. Feature engineering: create FamilySize = SibSp + Parch + 1.
4. Model: train a Random Forest classifier.
5. Evaluate using ROC AUC metric on validation set.
"""

AGENT_CODE = """ACTION: CODE
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
"""

AGENT_SUBMIT = "ACTION: FINAL_SUBMIT"


def main():
    registry = build_demo_registry()
    env = AutoMLEnvironment(
        registry,
        reward_weights=RewardWeights(performance=0.5, plan_coverage=0.2, code_quality=0.3),
        seed=42,
    )

    task_id = "titanic_binary"
    print("=" * 70)
    print(f"Starting episode for task: {task_id}")
    print("=" * 70)

    env.reset(task_id)

    obs = env.observe()
    print("\n--- OBSERVE ---")
    print(obs[:500], "..." if len(obs) > 500 else "")

    print("\n--- STEP 1: PLAN ---")
    out = env.step(AGENT_PLAN)
    print(f"reward={out.reward:.4f}, done={out.done}")
    print(out.state[:400])

    print("\n--- STEP 2: CODE ---")
    out = env.step(AGENT_CODE)
    print(f"reward={out.reward:.4f}, done={out.done}")
    print(out.state[:400])

    print("\n--- STEP 3: FINAL_SUBMIT ---")
    out = env.step(AGENT_SUBMIT)
    print(f"reward={out.reward:.4f}, done={out.done}")
    print(out.state)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
