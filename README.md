# Rapport de projet - Costa Verde Hotels

## Examen de rattrapage Machine Learning & Data Science - M2

ISPM - Madagascar ([www.ispm-edu.com](https://www.ispm-edu.com))


### 1. Informations sur l'étudiant

- **nom** : RAHERIMAHEFA
- **prénom(s)** : Vonimbola Zinot Fitiavana
- **classe** : IGGLIA5
- **numéro** : 45
- **rôle** : Data Scientist / Développeur ML (projet individuel)


### 2. Résumé du travail

- **Problématique et utilisation opérationnelle** :  
  Prédire les annulations de réservation dans les neuf communautés autonomes espagnoles.  
  L'objectif opérationnel est de permettre aux hôtels Costa Verde d'anticiper les annulations, de mieux gérer leurs stocks de chambres, et d'optimiser leur politique de surréservation.

- **Méthodologie** :
  - Analyse exploratoire des données (EDA)
  - Prétraitement : gestion des dates (extraction du mois, jour de semaine, délai), encodage des variables catégorielles (OneHotEncoder), normalisation des variables numériques (StandardScaler)
  - Validation temporelle (split chronologique 80/20)
  - Modèles testés : Régression Logistique, Random Forest, XGBoost
  - Sélection du meilleur modèle sur l'AUC-ROC

- **Résultat principal et limite** :  
  Le meilleur modèle est la **Régression Logistique** avec un **AUC-ROC = 0.6742** et une **accuracy = 0.6889**.  
  Limite : les données sont synthétiques, le déséquilibre des classes est marqué (74% / 26%), et l'AUC reste modéré ; une meilleure ingénierie des features pourrait améliorer la performance.

- **Cinq mots-clés** :  
  Machine Learning, Classification binaire, Annulation de réservation, Prédiction, Validation temporelle.


### 3. Contenu du dépôt et lien de présentation

- `zinot(1).ipynb` : notebook complet avec EDA, prétraitement, modélisation et génération de la soumission
- `submission.csv` : fichier de prédictions au format demandé
- `README.md` : ce rapport

**Versions des bibliothèques et instructions d'exécution :**
- Python 3.12
- pandas 2.2, numpy 1.26, scikit-learn 1.5, xgboost 2.0, matplotlib, seaborn
- Exécuter le notebook dans Google Colab 


### 4. Résultats sur la même période de validation

| Modèle               | Période d'ajustement | Période d'évaluation | AUC-ROC | Accuracy |
|----------------------|---------------------|----------------------|---------|----------|
| Baseline logistique  | 2023-01 – 2024-06   | 2024-07 – 2024-12    | 0.6742  | 0.6889   |
| Random Forest        | 2023-01 – 2024-06   | 2024-07 – 2024-12    | 0.6499  | 0.6877   |
| XGBoost              | 2023-01 – 2024-06   | 2024-07 – 2024-12    | 0.6194  | 0.6814   |
| Modèle final (LR)    | 2023-01 – 2024-12   | Test                 | —       | —        |

**Seuil retenu et période utilisée pour le choisir :**  
Seuil = 0.5, choisi sur la période de validation pour maximiser l'accuracy.

**Calibration (diagramme, observations et éventuelle méthode) :**  
La régression logistique produit des probabilités relativement bien calibrées. Un diagramme de fiabilité a été tracé : les probabilités prédites sont proches des fréquences observées pour la plupart des bins, sauf aux extrêmes. Une calibration de Platt pourrait être envisagée pour affiner.


### 5. Questions d'analyse

1. **Pourquoi l'accuracy peut-elle être trompeuse ici ? Quel apport ont F1 et PR-AUC ?**  
   L'accuracy est trompeuse car les classes sont déséquilibrées (environ 74% de non-annulations, 26% d'annulations). Un modèle qui prédit toujours "non annulé" aurait une accuracy de 74% mais ne détecterait aucune annulation. Le F1 et le PR-AUC tiennent compte du déséquilibre et évaluent mieux la capacité à détecter la classe minoritaire.

2. **Définissez un faux positif et un faux négatif pour l'hôtel. Comment choisir le seuil ?**  
   - Faux positif : prédire une annulation alors que la réservation est maintenue → perte de revenus potentiels (chambre bloquée à tort, surréservation inutile).  
   - Faux négatif : prédire une non-annulation alors que la réservation est annulée → surréservation non anticipée, client mécontent, perte de revenus.  
   Le seuil doit être choisi en fonction du coût relatif des deux erreurs. Si le coût d'une annulation non détectée est plus élevé, on abaisse le seuil (ex: 0.3).

3. **Quelles variables ou interactions créées ont aidé ? Quantifiez sur validation.**  
   Les variables dérivées des dates (mois, jour de la semaine, délai) et l'interaction `delai_reservation_jours × tarif_remboursable` ont amélioré l'AUC de 0.02 à 0.03 points sur la validation. La variable `delai_jours` (calculée comme la différence entre date d'arrivée et date de réservation) est particulièrement discriminante.

4. **Analysez au moins trois faux positifs et trois faux négatifs, sans exposer de réponses du test caché.**  

   Sur l'ensemble de validation (1601 observations), le modèle a produit :
   - **37 faux positifs** (réservations prédites comme annulées mais maintenues)
   - **461 faux négatifs** (réservations prédites comme maintenues mais annulées)

   **Faux positifs (FP) – prédit "annulé" alors que la réservation a été maintenue**

   | Cas | region_hotel | segment_client | delai_reservation_jours | tarif_remboursable | prix_moyen_nuit_eur | Probabilité prédite |
   |-----|--------------|----------------|-------------------------|--------------------|---------------------|---------------------|
   | FP1 | Catalogne | affaires | 63 | oui | 193.54 | 0.535 |
   | FP2 | Communauté valencienne | couple | 120 | oui | 113.58 | 0.599 |
   | FP3 | Aragon | famille | 106 | oui | 98.25 | 0.628 |

   **Interprétation :** Ces faux positifs concernent des clients avec un délai de réservation long (63 à 120 jours) et un tarif remboursable. Le modèle a appris que ces caractéristiques sont corrélées aux annulations, mais ici les clients ont maintenu leur séjour.

   **Faux négatifs (FN) – prédit "maintenu" alors que la réservation a été annulée**

   | Cas | region_hotel | segment_client | delai_reservation_jours | tarif_remboursable | prix_moyen_nuit_eur | Probabilité prédite |
   |-----|--------------|----------------|-------------------------|--------------------|---------------------|---------------------|
   | FN1 | Communauté valencienne | solo | 60 | oui | 180.74 | 0.413 |
   | FN2 | Aragon | famille | 32 | non | NaN | 0.187 |
   | FN3 | Aragon | famille | 12 | oui | 191.90 | 0.259 |

   **Interprétation :** Ces faux négatifs concernent des réservations qui semblaient "sûres" mais ont été annulées. La probabilité prédite est faible (0.19–0.41), ce qui signifie que le modèle était confiant à tort.

   **Statistiques descriptives :**

   | Statistique | Faux positifs (n=37) | Faux négatifs (n=461) |
   |-------------|----------------------|------------------------|
   | Délai moyen (jours) | 73.4 | 36.5 |
   | Prix moyen (€) | 177.8 | 201.8 |
   | Nuits moyen | 4.6 | 4.2 |
   | Probabilité moyenne | 0.556 | 0.108 |

   **Enseignements :** Le modèle est beaucoup plus performant pour prédire les maintiens que les annulations (461 FN contre 37 FP), à cause du déséquilibre des classes. Le seuil de 0.5 est trop élevé ; il faudrait le baisser à 0.3 pour détecter plus d'annulations.

5. **Quelle intervention préventive raisonnable recommandez-vous et sous quelles limites ?**  
   Recommandation : proposer une réduction ou un surclassement aux clients à risque élevé d'annulation (probabilité > 0.6) pour les inciter à maintenir leur réservation.  
   Limites : le modèle n'est pas parfait (461 FN), et une telle politique pourrait être perçue comme intrusive ou coûteuse si elle est mal ciblée. Une analyse coût-bénéfice est nécessaire.


### 6. Conclusion, reproductibilité et bibliographie

- **Conclusion et limites** :  
  Le modèle de régression logistique offre une première approche raisonnable pour prédire les annulations, avec un AUC de 0.67. Les limites incluent le déséquilibre des classes, la nature synthétique des données, et la nécessité d'une meilleure ingénierie des features.

- **Version Python / bibliothèques / graines** :  
  Python 3.12, pandas 2.2, scikit-learn 1.5, xgboost 2.0, numpy 1.26. Graine aléatoire : 42.

- **Procédure de reproduction** :  
  1. Télécharger les fichiers CSV depuis le dépôt GitHub (ou les charger dans Colab).  
  2. Ouvrir `zinot(1).ipynb` dans Colab.  
  3. Exécuter toutes les cellules dans l'ordre.  
  4. Le fichier `submission.csv` est généré à la fin.

- **Sources consultées** :  
  Documentation scikit-learn, documentation xgboost, cours ISPM Machine Learning.

- **Outils d'IA générative et contribution exacte, le cas échéant** :  
  Utilisation de Deepseek pour l'aide à la rédaction du code et du rapport.
