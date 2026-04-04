# default 8X enhancement
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut
# default 4X enhancement
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut --downbranch 2 --cropz 32 # (128 // 4)
# 8X enhancement with contrastive loss lbNCE=1
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --lbNCE 1


# filopodia on GHCL01
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_GHCL01 --env GHCL01 --nocut --downbranch 2 --cropz 32

# filopodia on GHCL03
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_GHCL03 --prj default/max10skip4 --env GHCL03 --nocut --downbranch 2 --cropz 32

# filopodia on NTHU_Gary3
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_NTHU_Gary3 --env NTHU_Gary3 --nocut --downbranch 2 --cropz 32

# mlflow ui --backend-store-uri "sqlite:////home/gary/workspace/logs/THX10SDM20xw/mlflow/mlflow.db" --port 5002  

# with mlflow
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanM0/max5skip4 --env brcb --dataset THX10SDM20xw --direction roiAdsp4 --lamb 5 --models vqcleanM0 --tracking_uri sqlite://///home/gary/workspace/logs/THX10SDM20xw/mlflow.db