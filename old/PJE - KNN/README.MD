# 🧠 KNN Tweet Classifier

A modular **K-Nearest Neighbors (KNN)** text classifier for **sentiment analysis** of tweets.
It works on **cleaned CSV datasets** and allows flexible configuration of:
- Distance metrics (Jaccard, Cosine, Levenshtein, etc.)
- Voting strategies (Majority / Weighted)
- Synonym normalization via JSON dictionary

---

## 🚀 Features

| Module | Description |
|---------|--------------|
| `distance.py` | Contains all text distance metrics (`Jaccard`, `Cosine`, `Levenshtein`, `Euclidean`, `Manhattan`) and the abstract base class `Distance`. Handles synonym normalization automatically. |
| `voting.py` | Defines how neighbors vote: `MajorityVote` (simple count) and `WeightedVote` (closer neighbors count more). |
| `synonym_mapper.py` | Loads a JSON file of synonym groups and replaces all words with a canonical synonym form. |
| `main.py` | End-to-end KNN evaluation script: loads dataset, splits train/test, fits model, and prints accuracy vs baseline. |

---

## 🧩 Project Structure

```
📦 knn-tweets
 ┣ 📜 distance.py
 ┣ 📜 voting.py
 ┣ 📜 synonym_mapper.py
 ┣ 📜 main.py
 ┣ 📜 synonyms.json
 ┗ 📜 test_cleaned_min.csv
```

---

## 📚 Data Format

Your CSV dataset should contain **two columns**:  
`label` and `tweet`, separated by comma or semicolon.

Example (`test_cleaned_min.csv`):
```csv
4,i love my kindle
0,this economy sucks
4,awesome new game release
2,not sure how i feel today
```

- `label`: integer (e.g. `0=negative`, `2=neutral`, `4=positive`)
- `tweet`: already cleaned lowercase text

---

## 🧠 Example `synonyms.json`

```json
{
  "0": ["football", "soccer", "foot"],
  "1": ["happy", "joyful", "glad", "content"],
  "2": ["angry", "mad", "furious"],
  "3": ["president", "obama", "biden"]
}
```

---

## ⚙️ How It Works

1. The classifier stores all training tweets and their labels.  
2. For each new tweet:
   - Compute the **distance** to all tweets in the training set.
   - Select the **k closest** neighbors.
   - Combine their labels via **majority** or **weighted** voting.
3. Output the predicted label.

---

## 🧩 Minimal Example

```python
from distance import JaccardDistance
from voting import MajorityVote
from knn_classifier import KNNClassifier
from synonym_mapper import SynonymMapper

synonym_mapper = SynonymMapper("synonyms.json")
distance = JaccardDistance(use_synonyms=True, synonym_json="synonyms.json")
voter = MajorityVote()

train_data = [
    (4, "i love my kindle"),
    (0, "i hate bad service"),
    (4, "the book is amazing"),
    (2, "it is okay not great")
]

knn = KNNClassifier(k=3, distance=distance, voter=voter)
knn.fit(train_data)

test_tweet = "i adore my book reader"
prediction = knn.predict_one(test_tweet)

print(f"Tweet: {test_tweet}")
print(f"Predicted label: {prediction}")
```

---

## 🧪 Evaluate on a Dataset

```bash
python main.py
```

Example output:
```
✅ Loaded 1200 tweets
→ Training on 840 | Testing on 360
📊 Label distribution:
Train: Counter({4: 540, 0: 180, 2: 120})
Test : Counter({4: 230, 0: 80, 2: 50})

✅ KNN Accuracy : 63.33% (95/150)
🪶 Baseline Acc : 32.67% (always predict '4')
```

---

## 🧰 Configuration

| Parameter | Location | Description |
|------------|-----------|--------------|
| `k` | `KNNClassifier(k=5)` | Number of nearest neighbors |
| `distance` | e.g. `JaccardDistance`, `CosineDistance` | Distance metric |
| `voter` | e.g. `MajorityVote`, `WeightedVote` | Voting method |
| `use_synonyms` | bool | Whether to apply synonym mapping |
| `synonym_json` | path | JSON file for synonyms |

---

## 🧪 Requirements



Install everything with:
```bash
pip install -r requirements.txt
```

---
