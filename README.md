# 🤖 Classification automatique des Biens de Consommation

**Classification automatique de produits e-commerce à partir de descriptions textuelles et d’images.**  
Projet de Data Science visant à développer un moteur d’attribution de catégories capable d’identifier automatiquement le type de produit à partir de ses métadonnées.

---

## 🎯 Objectif du projet

L’objectif de ce projet est de **concevoir un modèle de classification multi-modale** (texte + image) permettant de prédire la catégorie de produits sur la base :
- des **titres et descriptions textuelles**,  
- et des **images associées** (dans un second temps).

Ce travail s’inscrit dans une logique d’**industrialisation** d’un moteur de classification automatique utilisable sur un site e-commerce ou un catalogue produit à grande échelle.

---

## 🧩 Contenu du dépôt

Le dépôt contient :

- **Notebooks Jupyter** :
  1. `preprocessing_feature_extraction.ipynb`  
     → Nettoyage, normalisation et vectorisation des textes (BoW, TF-IDF, Word2Vec).  
  2. `text_classification_models.ipynb`  
     → Comparaison de modèles de NLP : TF-IDF, Word2Vec, USE, BERT.  
  3. `image_classification_models.ipynb`  
     → Extraction d’attributs visuels (SIFT, ORB) et classification CNN (VGG16, ResNet50).  

- **Dataset** :  
  `flipkart_com-ecommerce_sample_1050.csv` – extrait de la base *Flipkart e-commerce product sample*.  

- **Présentation PowerPoint** :  
  `Martineau_Alexandre_4_presentation_102024.pdf` – synthèse visuelle du projet et comparaison des performances modèles.

---

## 🧠 Méthodologie

### 1. Préparation & nettoyage
- Vérification des valeurs manquantes, doublons et anomalies.  
- Séparation du corpus textuel et des métadonnées.  
- Normalisation linguistique : suppression de la ponctuation, *lemmatisation*, *stopwords*, *lowercase*.  

### 2. Feature Engineering
- **Texte** :
  - Bag of Words (baseline)
  - TF-IDF (pondération contextuelle)
  - Word2Vec (GoogleNews-300)
  - Universal Sentence Encoder (USE)
  - BERT (Hugging Face / TensorFlow)
- **Image** :
  - Descripteurs SIFT / ORB
  - CNN pré-entraînés : *VGG16*, *ResNet50*

### 3. Modélisation
- **Classifieurs testés** :
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - SVM linéaire et radial
  - Réseaux de neurones profonds (Dense / CNN)
- **Évaluation** :
  - Accuracy, Precision, Recall, F1-score  
  - Confusion matrix, classification report  
  - ARI (*Adjusted Rand Index*) et courbes ROC

---

## 📊 Résultats principaux

| Modèle | Type | Accuracy | ARI | Commentaire |
|---------|------|-----------|-----|--------------|
| Bag of Words | Texte | **0.93** | 0.58 | Baseline robuste |
| TF-IDF | Texte | **0.95** | 0.62 | Meilleur compromis performance / coût |
| Word2Vec | Texte | 0.72 | 0.30 | Pertinent pour corpus long |
| USE | Texte | 0.89 | 0.50 | Bon contexte sémantique |
| BERT (Hugging Face) | Texte | **0.93** | 0.60 | Excellente compréhension contextuelle |
| CNN (VGG16 / ResNet50) | Image | 0.85 | - | Classification visuelle efficace |

> 🏆 **Meilleure performance globale** : TF-IDF + Gradient Boosting (95 % accuracy)  
> BERT offre cependant une meilleure compréhension linguistique pour de futures extensions.

---

## 💡 Insights clés

- Le **texte** contient une forte capacité discriminante → les descriptions produits suffisent à atteindre >90 % de précision.  
- Les **images** complètent utilement la catégorisation mais nécessitent davantage de ressources GPU.  
- Le **modèle BERT** est le plus prometteur pour un usage en production NLP (API de classification automatisée).  

---

## 🛠️ Technologies utilisées

- **Langage** : Python 3.11  
- **Librairies principales** :  
  `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `transformers`, `nltk`, `gensim`, `opencv`, `matplotlib`, `seaborn`
- **Outils NLP** : TF-IDF, Word2Vec, USE, BERT  
- **Vision par ordinateur** : SIFT, ORB, CNN (VGG16, ResNet50)  
- **Évaluation & Visualisation** : confusion matrix, t-SNE, PCA, heatmaps

---

## 📂 Structure du dépôt

```text
classification_automatique
│
├── data/
│   └── flipkart_com-ecommerce_sample_1050.csv
│
├── notebooks/
│   ├── preprocessing_feature_extraction.ipynb
│   ├── text_classification_models.ipynb
│   ├── image_classification_models.ipynb
│
├── Martineau_Alexandre_4_presentation_102024.pdf
└── README.md
```

---

## 🚀 Perspectives d’amélioration

- Intégration d’une API Flask/FastAPI pour la classification en temps réel.
- Fusion texte + image dans un modèle multimodal (BERT + CNN).
- Optimisation des embeddings via fine-tuning.
- Implémentation d’un système de catégorisation hiérarchique (sous-catégories).

---

## 🧩 Conclusion
Ce projet démontre la faisabilité d’un système de classification automatique multi-modale performant sur des données e-commerce réelles.
Les résultats confirment la pertinence d’une approche hybride combinant NLP (BERT/TF-IDF) et vision (CNN) pour automatiser la catégorisation produit à grande échelle.
