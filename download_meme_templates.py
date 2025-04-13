from download_meme_images import download_image
from pathlib import Path
from tqdm.auto import tqdm
import json

template_path = Path("data/meme_template.json")
template_img_path = Path("data/templates")
template_dict = json.load(template_path.open())
template_img_path.mkdir(exist_ok=True)

for name, (url, id) in tqdm(template_dict.items()):
    normalized_name = name.lower().replace("/", " ").replace(" ", "_")
    file_name = f"{id}-{normalized_name}"
    file_extension = url.split(".")[-1]
    file_path = template_img_path / f"{file_name}.{file_extension}"

    if not file_path.exists():
        try:
            download_image(url, file_path)
        except Exception as e:
            continue