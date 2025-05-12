# csc8608

---

## Aperçu

1. **Ingestion**  
   Téléchargement automatique du dataset MVTec pour la classe choisie.

2. **Prétraitement & Split**  
   Construction des objets PyTorch `Dataset` (`train` / `test`) via `MVTecDataset`.

3. **Entraînement & Évaluation**  
   - Fit de PatchCore (sélection de coreset, anomalie score).  
   - Calcul du ROC-AUC image / pixel.

4. **Tracking & Versioning**  
   - Enregistrement des hyperparamètres, métriques et du modèle avec MLflow.  
   - Sauvegarde de la memory-bank pour réutilisation (`use_cache=True`).

5. **Orchestration**  
   Un DAG Airflow à deux tâches :
   1. `ingest` → `src/ingest.py`  
   2. `train_and_log` → `src/train_mlflow.py`

---

## Structure du projet 

```text
csc8608/
├── airflow_home/         # Base Airflow (DB, logs, etc.)
├── dags/                 # définitions des DAGs Airflow
│   └── patchcore_full_pipeline.py
├── data/
│   ├── data.py
├── datasets/             # datasets téléchargés (par ingest)
├── model/
│   └── patch_core.py     # implémentation PatchCore
├── src/
│   ├── ingest.py         # tâche Airflow d’ingestion/prétraitement
│   └── train_mlflow.py   # tâche Airflow d’entraînement + MLflow
├── utils/
│   ├── utils.py
├── mlruns/               # répertoire MLflow Tracking
├── patchcore_cache/      # memory-banks sérialisées
├── requirements.txt
└── README.md 

```


## 🛠️ Installation

```pip install -r requirements.txt

# Terminal 1 : Import de Airflow
# Bien préciser le chemin qui contient le DAG ( ./dags)
export AIRFLOW_HOME=$PWD/airflow_home
airflow db init

# Terminal 1: Airflow Webserver + Scheduler
airflow webserver --port 8080 &
airflow scheduler &

# Terminal 2: MLflow UI
mlflow ui --port 5000 &
