## Installation et environnement

Ce projet utilise un environnement virtuel Python (`venv`) pour isoler ses dépendances et éviter tout conflit avec le système global.

### Prérequis
- Python 3.10+

### Mise en place

1. **Créer l'environnement virtuel :**
   ```bash
   python3 -m venv .venv


2. choix du kernel :
3. Un kernel est le moteur qui fait tourner le code Python dans ton notebook.

Sans lui, le fichier .ipynb n'est qu'un document texte inerte :

L'interface (VS Code) t'affiche les cellules.

Quand tu cliques sur Run / Exécuter, elle envoie le code au kernel.

Le kernel exécute le script avec les bibliothèques installées dans ton .venv et renvoie le résultat (texte, erreurs, graphiques) à l'écran.

Sélectionner le bon kernel permet simplement à VS Code de savoir quelle version de Python et quels paquets (langchain, pypdf, etc.) utiliser pour faire tourner tes cellules.