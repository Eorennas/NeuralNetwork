from pycocotools.coco import COCO
import os, shutil

# Caminhos
annFile = "coco/annotations/instances_train2017.json"
imgDir  = "coco/train2017/"

# Criar pastas destino
os.makedirs("dataset/pessoa", exist_ok=True)
os.makedirs("dataset/nao_pessoa", exist_ok=True)

# Carregar COCO
coco = COCO(annFile)

# IDs de imagens com pessoa
catIds = coco.getCatIds(catNms=['person'])
imgIds_com_pessoa = coco.getImgIds(catIds=catIds)

# Todas as imagens do dataset
todos_ids = coco.getImgIds()

# IDs sem pessoa
imgIds_sem_pessoa = list(set(todos_ids) - set(imgIds_com_pessoa))

# --- Copiar alguns exemplos ---
import random
random.shuffle(imgIds_com_pessoa)
random.shuffle(imgIds_sem_pessoa)

for img_id in imgIds_com_pessoa[:2000]:
    img_info = coco.loadImgs(img_id)[0]
    src = os.path.join(imgDir, img_info['file_name'])
    dst = os.path.join("dataset/pessoa", img_info['file_name'])
    shutil.copy(src, dst)

for img_id in imgIds_sem_pessoa[:2000]:
    img_info = coco.loadImgs(img_id)[0]
    src = os.path.join(imgDir, img_info['file_name'])
    dst = os.path.join("dataset/nao_pessoa", img_info['file_name'])
    shutil.copy(src, dst)
