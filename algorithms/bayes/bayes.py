import csv
import math
from collections import defaultdict
import os

# Codes standards pour l'affichage
CLASS_LABELS = {
    0: "negatif",
    2: "neutre",
    4: "positif"
}


class NaiveBayesClassifier:


    def __init__(self, smoothing=1.0, use_binary=False, n_gram=1):

        self.smoothing = smoothing
        self.use_binary = use_binary
        self.n_gram = n_gram

        self.prior = {}  # Probabilité de chaque classe P(c)
        self.cond_prob = {}  # Probabilité conditionnelle P(mot | classe)
        self.vocab = set()  # Vocabulaire complet
        self.classes = set()  # Ensemble des classes (0, 2, 4)

    def _get_ngrams(self, text):
        if not isinstance(text, str):
            return []

        # Tokenisation simple : minuscule + découpage par espace
        tokens = text.lower().split()

        if self.n_gram == 1:
            # Cas classique : juste les mots
            return tokens

        elif self.n_gram == 2:
            # Bigrammes uniquement (paires de mots consécutifs)
            if len(tokens) < 2:
                return []
            return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

        elif self.n_gram == 3:
            # Unigrammes + Bigrammes (Mots simples ET paires)
            unigrams = tokens
            bigrams = []
            if len(tokens) >= 2:
                bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
            return unigrams + bigrams

        return tokens

    def fit(self, data):

        self.classes = set()
        self.vocab = set()

        # 1. Nettoyage et filtrage des données
        clean_data = []
        if not data:
            return

        for item in data:
            # Robustesse : on s'attend à un tuple (label, tweet)
            if len(item) < 2:
                continue

            try:
                # On s'assure que le label est un entier
                label = int(item[0])
                tweet = str(item[1])
            except:
                continue

            # On ne garde que les labels valides du projet (0, 2, 4)
            if label in [0, 2, 4]:
                clean_data.append((label, tweet))
                self.classes.add(label)

        if not clean_data:
            print("⚠️ Aucune donnée d'entraînement valide.")
            return

        # 2. Initialisation des compteurs
        # word_count[classe][mot] = nombre d'occurrences
        word_count = {c: defaultdict(int) for c in self.classes}
        # total_words[classe] = nombre total de mots (ou tokens) dans la classe
        total_words = {c: 0 for c in self.classes}
        # class_count[classe] = nombre de tweets de cette classe
        class_count = defaultdict(int)

        # 3. Comptage (Training)
        for label, tweet in clean_data:
            class_count[label] += 1
            tokens = self._get_ngrams(tweet)

            if self.use_binary:
                tokens = set(tokens)

            for w in tokens:
                self.vocab.add(w)
                word_count[label][w] += 1
                total_words[label] += 1

        # 4. Calcul des probabilités
        V = len(self.vocab)
        total_tweets = len(clean_data)

        self.prior = {c: class_count[c] / total_tweets for c in self.classes}

        self.cond_prob = {}
        for c in self.classes:
            self.cond_prob[c] = {}
            denom = total_words[c] + (self.smoothing * V)

            for w in self.vocab:
                count = word_count[c][w]
                self.cond_prob[c][w] = (count + self.smoothing) / denom

        print(f"✅ Modèle entraîné sur {total_tweets} tweets (Vocab: {V} tokens).")

    def predict_one(self, text):
        # Si le modèle n'est pas entraîné, on retourne Neutre par défaut
        if not self.prior:
            return 2

        tokens = self._get_ngrams(text)

        # --- Gestion des mots inconnus (IMPORTANT) ---
        # On ne garde que les mots qui sont dans le vocabulaire appris
        mots_connus = [w for w in tokens if w in self.vocab]

        # Si aucun mot n'est connu, le modèle ne peut pas décider -> Neutre
        if not mots_connus:
            return 2

        scores = {}
        for c in self.prior:
            # Score initial = Log(P(Classe))
            score = math.log(self.prior[c])

            for w in mots_connus:
                # On ajoute Log(P(Mot | Classe))
                # On est sûr que w existe dans cond_prob car w est dans vocab
                score += math.log(self.cond_prob[c][w])

            scores[c] = score

        # Retourne la classe avec le score le plus élevé (max log-prob)
        return max(scores, key=scores.get)

    def predict_batch(self, texts):
        """Prédiction pour une liste de tweets."""
        return [self.predict_one(t) for t in texts]




def load_training_csv_legacy(path):
    """Ancien chargeur CSV simple (gardé pour compatibilité)"""
    data = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row and len(row) >= 2:
                    try:
                        data.append((int(row[0]), row[1]))
                    except:
                        continue
    except FileNotFoundError:
        pass
    return data


def load_test_csv_legacy(path):
    rows = []
    header = None
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row: rows.append(row)
    except:
        pass
    return rows, header


def save_predictions_legacy(path, rows, header, predictions):
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            new_header = (header + ["pred_num", "pred_txt"]) if header else ["pred_num", "pred_txt"]
            writer.writerow(new_header)
            for row, pred in zip(rows, predictions):
                writer.writerow(row + [pred, CLASS_LABELS.get(pred, "?")])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    nb = NaiveBayesClassifier()
    data = [(4, "I love this app"), (0, "This is bad"), (2, "It is okay")]
    nb.fit(data)
    print("Test:", nb.predict_one("I love it"))