from tqdm import tqdm
from pathlib import Path
import cv2

raw_image = Path("./data/DUTS-TE/Original_Img")
export_folder = "./data/my_model_result/"
img_list = list(raw_image.glob("**/*"))

pbar = tqdm(img_list)
for img_path in img_list:
    
    image = cv2.imread(str(img_path))
    blurred = cv2.GaussianBlur(image, (13,13), 0)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cv2.imwrite(export_folder + img_path.name, threshMap)
    pbar.update(1)