# 🧠 TP - Annotation de Tweets par Mots Clés

## 📘 Description du Projet

Ce projet implémente un **système d’annotation automatique** de tweets basé sur des mots-clés positifs et négatifs.  
L’objectif est de déterminer la **polarité** d’un tweet selon la grammaire suivante :

| Polarité  | Valeur |
|------------|--------|
| Négatif    | 0      |
| Neutre     | 2      |
| Positif    | 4      |

Le projet est conçu de manière **modulaire** pour être réutilisé dans d'autres contextes (GUI, API, etc.).  
Chaque module a une responsabilité claire : lecture, annotation, évaluation, etc.

---

## 🧩 Structure du Projet

```
📦 Projet Annotation
│
├── annotation/
│   ├── __init__.py
│   ├── annotation.py        # Contient la logique de base : AnnotationParams + annotate_tweet()
│   ├── keyword_annotator.py # Gère l’annotation d’un CSV complet (add / override)
│
├── evaluate.py              # Évalue la précision entre labels réels et prédits
├── words.py                 # Lecture concurrente des fichiers de mots positifs/négatifs
├── keywords.json            # Fichier exporté contenant les listes de mots
├── test.csv                 # Exemple de dataset nettoyé (label, tweet)
├── main.py                  # Exemple de script d'exécution
└── README.md                # Documentation
```

---

## ⚙️ Étape 1 : Préparation des Mots-Clés

Avant d’annoter les tweets, on doit extraire les mots positifs et négatifs depuis deux fichiers texte :
`positive.txt` et `negative.txt`.

Le script suivant (dans **words.py**) lit ces fichiers et crée un fichier JSON utilisable :

```python
from words import load_keywords

# Crée keywords.json à partir des fichiers texte
load_keywords()
```

Cela génère un fichier `keywords.json` de la forme :
```json
{
  "positive": ["love", "happy", "great", "amazing", ...],
  "negative": ["bad", "hate", "sad", "awful", ...]
}
```

---

## 🧠 Étape 2 : Le Cœur du Système (annotation.py)

Ce module contient :

### `AnnotationParams`
Un conteneur d’informations (OOP) qui stocke :  
- les mots positifs/négatifs,  
- un cache (mémoïsation),  
- des paramètres de configuration (minuscule, stemming, etc.).

### `annotate_tweet(tweet, params)`
Applique les règles suivantes :  
- Si le tweet contient plus de mots positifs → 4  
- Plus de mots négatifs → 0  
- Autant ou aucun → 2

Exemple d’utilisation :
```python
from annotation import AnnotationParams, annotate_tweet

params = AnnotationParams(positive_words=["love", "great"], negative_words=["hate", "bad"])
tweet = "I love this phone but hate the battery"
label = annotate_tweet(tweet, params)
print(label)  # Résultat: 2 (autant de mots positifs que négatifs)
```

---

## 🧰 Étape 3 : Annotation d’un Dataset (keyword_annotator.py)

Le module `KeywordAnnotator` permet d’appliquer la logique sur un **CSV complet**.

Deux modes disponibles :
- `"override"` → remplace les valeurs de la colonne de label existante
- `"add"` → ajoute une nouvelle colonne `predicted_label` (et garde l’ancienne)

Exemple :
```python
from annotation.keyword_annotator import KeywordAnnotator

annotator = KeywordAnnotator(json_path="keywords.json")
df = annotator.annotate("test.csv", "annotated.csv", mode="add")
```

---

## 📈 Étape 4 : Évaluation (evaluate.py)

Permet de comparer les labels réels et prédits :

```python
from evaluate import evaluate_accuracy

evaluate_accuracy(df, true_col="0", pred_col="predicted_label")
```

Affiche :
```
✅ Accuracy: 63.72%  (912/1431 correct)
```

---

## 🚀 Étape 5 : Exemple Complet (main.py)

```python
from annotation.keyword_annotator import KeywordAnnotator
from evaluate import evaluate_accuracy

# 1️⃣ Annoter le dataset
annotator = KeywordAnnotator("keywords.json")
df = annotator.annotate("test.csv", "annotated.csv", mode="add")

# 2️⃣ Évaluer la précision
evaluate_accuracy(df, true_col="0", pred_col="predicted_label")
```

---

## 🧠 Recommandations
- Toujours utiliser un **fichier nettoyé** (issu du TP1) pour de meilleurs résultats.
- Éviter les tweets multilingues sans nettoyage.
- En cas de grands volumes, utiliser un dictionnaire optimisé (hash lookup).

---

© 2025 — Projet d’Annotation de Tweets — M1 Informatique, Université de Lille
