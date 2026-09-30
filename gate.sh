#!/bin/bash
set -e
THRESHOLD=${1:-0.85}
JOB=$(az ml job create -f pipeline.yml --query name -o tsv)
echo "Pipeline job: $JOB"
az ml job stream -n $JOB || true
STATUS=$(az ml job show -n $JOB --query status -o tsv)
[ "$STATUS" = "Completed" ] || { echo "Pipeline ended with status: $STATUS"; exit 1; }
TRAIN=$(az ml job list --parent-job-name $JOB --query "[?display_name=='train'].name | [0]" -o tsv)
export MLFLOW_TRACKING_URI=$(az ml workspace show --query mlflow_tracking_uri -o tsv)
ACC=$(python3 -c "import mlflow;print(mlflow.tracking.MlflowClient().get_run('$TRAIN').data.metrics['accuracy'])")
echo "accuracy=$ACC threshold=$THRESHOLD"
if python3 -c "import sys;sys.exit(0 if float('$ACC')>=float('$THRESHOLD') else 1)"; then
  az ml model create --name iris-model --type mlflow_model \
    --path azureml://jobs/$TRAIN/outputs/model --tags accuracy=$ACC
  echo "GATE PASSED: model registered"
else
  echo "GATE FAILED: model NOT registered"; exit 1
fi
