Ce dossier contient les résultats d'entraînement, de test, de validation et les métriques d'évaluation.

Fichiers générés par les scripts:
- train_results.json: Résultats d'entraînement
- test_results.json: Résultats de test
- val_results.json: Résultats de validation
- metrics.json: Métriques d'évaluation

Ces fichiers sont générés automatiquement lorsque vous exécutez les commandes suivantes avec l'option --save:

python show_results.py --type train --save
python show_results.py --type test --save
python show_results.py --type val --save
python show_metrics.py --save
