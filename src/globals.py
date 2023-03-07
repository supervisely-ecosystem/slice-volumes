import os
import supervisely as sly
from dotenv import load_dotenv


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

AXIS_NAME = {
    "x": "Sagittal",
    "y": "Coronal",
    "z": "Axial",
}

PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id(raise_not_found=False)
STORAGE_DIR = sly.app.get_data_dir()
sly.fs.clean_dir(STORAGE_DIR)
key_id_map = None

api: sly.Api = sly.Api.from_env()

project_info = api.project.get_info_by_id(PROJECT_ID)
project_meta_json = api.project.get_meta(PROJECT_ID)
project_meta = sly.ProjectMeta.from_json(project_meta_json)

if DATASET_ID is not None:
    datasets = [api.dataset.get_info_by_id(DATASET_ID)]
else:
    datasets = api.dataset.get_list(PROJECT_ID)

volumes = []
volume_parent = {}
for ds_idx, ds in enumerate(datasets):
    for volume in api.volume.get_list(ds.id):
        volumes.append(volume)
        volume_parent[volume.id] = ds_idx

selected_volume_idx = None


obj_classes_data = api.object_class.get_list(PROJECT_ID)
for i in range(len(obj_classes_data)):
    obj_classes_data[i] = {
        "id": obj_classes_data[i].id,
        "title": obj_classes_data[i].name,
        "shape": obj_classes_data[i].shape,
        "color": obj_classes_data[i].color,
    }
obj_classes = {info["id"]: sly.ObjClass.from_json(info) for info in obj_classes_data}
