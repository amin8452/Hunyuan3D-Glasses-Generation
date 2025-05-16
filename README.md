# Hunyuan3D Glasses Generation

Générez des modèles 3D de lunettes à partir d'images 2D en utilisant l'API Hunyuan3D.

## Installation

```bash
pip install -r requirements.txt
```

## Commandes Disponibles

### Génération de modèles 3D

Pour générer des modèles 3D de lunettes à partir de l'API:

```bash
python generate_glasses.py --api
```

Options:
- `--count <nombre>`: Spécifie le nombre d'images à récupérer (défaut: 5)

Exemple:
```bash
python generate_glasses.py --api --count 10
```

### Afficher les résultats d'entraînement

```bash
python show_results.py --type train
```

Options:
- `--save`: Sauvegarde les résultats dans un fichier JSON

### Afficher les résultats de test

```bash
python show_results.py --type test
```

Options:
- `--save`: Sauvegarde les résultats dans un fichier JSON

### Afficher les résultats de validation

```bash
python show_results.py --type val
```

Options:
- `--save`: Sauvegarde les résultats dans un fichier JSON

### Afficher les métriques d'évaluation

```bash
python show_metrics.py
```

Options:
- `--detailed`: Affiche des métriques plus détaillées
- `--save`: Sauvegarde les métriques dans un fichier JSON

Exemple:
```bash
python show_metrics.py --detailed --save
```

## Structure des Dossiers

- `output/`: Contient les modèles 3D générés
- `results/`: Contient les résultats d'entraînement, de test, de validation et les métriques

## Données

Toutes les données sont récupérées automatiquement depuis l'API Hunyuan3D. Aucune image locale n'est nécessaire.

## Remarques

- Les résultats d'entraînement, de test et de validation sont récupérés depuis l'API
- Les métriques d'évaluation sont calculées sur les modèles générés
- Tous les fichiers générés sont sauvegardés dans les dossiers appropriés
