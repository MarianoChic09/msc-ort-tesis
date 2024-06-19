

$project = "msc-ort-tesis"
# $project = "llama-index-property-graph-testing"

$Version = "0.1.0"
# -------------------------- Run locally ----------------------------
cd $project

micromamba create -n $project -y -c conda-forge python=3.10 

micromamba activate $project

pip install ipykernel

pip install -r requirements.txt


uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload 

# -------------------------- Docker Build and run ----------------------------
docker build -t "$($project):$($Version)" .

docker tag "$($project):$($Version)" synsmartacr.azurecr.io/"$($project):$($Version)"

docker push synsmartacr.azurecr.io/"$($project):$($Version)"

docker run --env-file ./.env -t "$($project):$($Version)"

docker run --env-file ./.env -t "$($project):$($Version)" ls /app

