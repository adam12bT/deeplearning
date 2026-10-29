"""Start a local MLflow dashboard on the same SQLite store used by training."""
import subprocess
import sys
from mlops.tracking import ROOT,local_tracking_uri

if __name__=='__main__':
    artifacts=ROOT/'artifacts/mlflow/files';artifacts.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','mlflow','server','--backend-store-uri',local_tracking_uri(),
        '--default-artifact-root',artifacts.as_uri(),'--host','127.0.0.1','--port','5000','--workers','1'],check=True)
