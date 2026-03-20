import pandas as pd
import numpy as np
import random
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

TYPE_TO_COL = {
    "bug": "against_bug", "dark": "against_dark", "dragon": "against_dragon",
    "electric": "against_electric", "fairy": "against_fairy",
    "fighting": "against_fight",  
    "fire": "against_fire", "flying": "against_flying", "ghost": "against_ghost",
    "grass": "against_grass", "ground": "against_ground", "ice": "against_ice",
    "normal": "against_normal", "poison": "against_poison", "psychic": "against_psychic",
    "rock": "against_rock", "steel": "against_steel", "water": "against_water",
}

def get_multiplier(row, attack_type: str) -> float:
    col = TYPE_TO_COL.get(attack_type)
    if col and col in row.index:
        return float(row[col])
    return 1.0

def calc_damage(atk_stat, def_stat, multiplier) -> float:
    BASE_POWER = 40
    rand = random.uniform(0.85, 1.0)
    return BASE_POWER * (atk_stat / max(def_stat, 1)) * multiplier * rand


#Load data
df = pd.read_csv('pokemon.csv')
df["type2"] = df["type2"].fillna(df["type1"])
df = df.reset_index(drop=True)

print(f"Loaded {len(df)} Pokemon.")


#Simulate battles to build training data
records = []
NUM_SIMULATIONS = 10_000 

for _ in range(NUM_SIMULATIONS):
    # pick two random pokemon
    ai_row     = df.sample(1).iloc[0]
    player_row = df.sample(1).iloc[0]

    # random HP states (simulate mid-battle moments, not just full health)
    ai_hp_ratio     = random.uniform(0.1, 1.0)
    player_hp_ratio = random.uniform(0.1, 1.0)

    # calculate multipliers for each AI attack option
    mult1 = get_multiplier(player_row, ai_row["type1"])
    mult2 = get_multiplier(player_row, ai_row["type2"])

    # simulate damage for each choice
    dmg1 = np.mean([calc_damage(ai_row["attack"], player_row["defense"], mult1) for _ in range(3)])
    dmg2 = np.mean([calc_damage(ai_row["attack"], player_row["defense"], mult2) for _ in range(3)])

    # label: 0 if type1 is better, 1 if type2 is better
    # if equal (same type1/type2), randomly label — teaches the model to handle ties
    if dmg1 > dmg2:
        label = 0
    elif dmg2 > dmg1:
        label = 1
    else:
        label = random.randint(0, 1)

    records.append({
        "ai_hp_ratio": ai_hp_ratio,
        "ai_attack": ai_row["attack"],
        "ai_sp_attack": ai_row["sp_attack"],
        "ai_speed": ai_row["speed"],
        "player_hp_ratio": player_hp_ratio,
        "player_defense": player_row["defense"],
        "player_sp_defense": player_row["sp_defense"],
        "mult_type1": mult1,
        "mult_type2": mult2,
        "label": label,
    })

train_df = pd.DataFrame(records)
print(f"\nGenerated {len(train_df)} training samples.")
print(f"Label distribution:\n{train_df['label'].value_counts()}\n")


#Train_(finally)
FEATURES = [
    "ai_hp_ratio", "ai_attack", "ai_sp_attack", "ai_speed",
    "player_hp_ratio", "player_defense", "player_sp_defense",
    "mult_type1", "mult_type2",
]

X = train_df[FEATURES].values
y = train_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["type1", "type2"]))

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Feature importances:")
print(importances.round(3))


#Save model
joblib.dump(model, "ai_model.pkl")
print("\n✅ Model saved as ai_model.pkl")
print("   Put ai_model.pkl in the same folder as pokemon_game.py")
