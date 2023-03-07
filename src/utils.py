import math
import os
from pathlib import Path
import nrrd
import numpy as np
import cv2
import trimesh
import supervisely as sly
import src.globals as g


def read_volume(volume):
    filepath = g.STORAGE_DIR + volume.name
    if not Path(filepath).exists():
        g.api.volume.download_path(volume.id, filepath)

    data, meta = sly.volume.read_nrrd_serie_volume_np(filepath)
    return data, meta


def get_volume_annotation(volume_id):
    vol_ann_json = g.api.volume.annotation.download(volume_id)
    vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, g.project_meta, g.key_id_map)
    return vol_ann


def download_spatial_figures(volume_ann):
    id_to_paths = {}
    for sp_figure in volume_ann.spatial_figures:
        figure_id = g.key_id_map.get_figure_id(sp_figure.key())
        id_to_paths[figure_id] = f"{g.STORAGE_DIR}/{figure_id}.stl"
    if id_to_paths:
        g.api.volume.figure.download_stl_meshes(*zip(*id_to_paths.items()))
    return id_to_paths


def matrix_from_volume_meta(meta):
    try:
        space_directions = meta["spacing"]
        space_origin = meta["origin"]
    except KeyError as e:
        raise IOError(
            'Need the meta "{}" field to determine the mapping from voxels to world coordinates.'.format(
                e
            )
        )

    # "... the space directions field gives, one column at a time, the mapping from image space to world space
    # coordinates ... [1]_" -> list of columns, needs to be transposed
    trans_3x3 = np.array(space_directions).T
    trans_4x4 = np.eye(4)
    trans_4x4[:3, :3] *= trans_3x3
    trans_4x4[:3, 3] = space_origin

    return trans_4x4


# def mask_from_stl(stl_path, mask_shape):
def mask_from_stl(mask_shape, voxel_to_world, stl_path):
    world_to_voxel = np.linalg.inv(voxel_to_world)

    mesh = trimesh.load(stl_path)

    min_vec = [float("inf"), float("inf"), float("inf")]
    max_vec = [float("-inf"), float("-inf"), float("-inf")]

    mesh.apply_scale((-1, -1, 1))  # LPS to RAS
    mesh.apply_transform(world_to_voxel)

    for vert in mesh.vertices:
        min_vec[0] = min(min_vec[0], vert[0])
        min_vec[1] = min(min_vec[1], vert[1])
        min_vec[2] = min(min_vec[2], vert[2])

        max_vec[0] = max(max_vec[0], vert[0])
        max_vec[1] = max(max_vec[1], vert[1])
        max_vec[2] = max(max_vec[2], vert[2])

    center = [(min_v + max_v) / 2 for min_v, max_v in zip(min_vec, max_vec)]

    try:
        voxel = mesh.voxelized(pitch=1.0)
    except Exception as e:
        sly.logger.error(e)
        sly.logger.warning(
            "Couldn't voxelize file {!r}".format(".".join(stl_path.split(".")[-2:])),
            extra={"file_path": stl_path},
        )
        return np.zeros(mask_shape).astype(np.bool)

    voxel = voxel.fill()
    mask = voxel.matrix.astype(np.bool)
    padded_mask = np.zeros(mask_shape).astype(np.bool)

    # find dimension coords
    start = [
        math.ceil(center_v - shape_v / 2)
        for center_v, shape_v in zip(center, mask.shape)
    ]
    end = [
        math.ceil(center_v + shape_v / 2)
        for center_v, shape_v in zip(center, mask.shape)
    ]

    # find intersections
    vol_inter_max = [max(start[0], 0), max(start[1], 0), max(start[2], 0)]
    vol_inter_min = [
        min(end[0], mask_shape[0]),
        min(end[1], mask_shape[1]),
        min(end[2], mask_shape[2]),
    ]

    padded_mask[
        vol_inter_max[0] : vol_inter_min[0],
        vol_inter_max[1] : vol_inter_min[1],
        vol_inter_max[2] : vol_inter_min[2],
    ] = mask[
        vol_inter_max[0] - start[0] : vol_inter_min[0] - start[0],
        vol_inter_max[1] - start[1] : vol_inter_min[1] - start[1],
        vol_inter_max[2] - start[2] : vol_inter_min[2] - start[2],
    ]

    return padded_mask


def get_spatial_masks(volume_ann: sly.VolumeAnnotation, volume_meta, data_shape):
    spatial_figures_masks = {}
    nrrd_matrix = matrix_from_volume_meta(volume_meta)
    spatial_figures_paths = download_spatial_figures(volume_ann)
    for sp_figure in volume_ann.spatial_figures:
        figure_id = g.key_id_map.get_figure_id(sp_figure.key())
        stl_path = spatial_figures_paths[figure_id]
        mask = mask_from_stl(data_shape, nrrd_matrix, stl_path)
        spatial_figures_masks[figure_id] = mask
    return spatial_figures_masks


def get_max_frame_n(data, axis):
    if axis == "x":
        return data.shape[0]
    if axis == "y":
        return data.shape[1]
    if axis == "z":
        return data.shape[2]


def get_target_size(spacing, shape, axis):
    if axis == "x":
        multiplier = 1 / min(spacing[1], spacing[2])
        dsize = (
            int(multiplier * spacing[1] * shape[1]),
            int(multiplier * spacing[2] * shape[2]),
        )
    if axis == "y":
        multiplier = 1 / min(spacing[0], spacing[2])
        dsize = (
            int(multiplier * spacing[0] * shape[0]),
            int(multiplier * spacing[2] * shape[2]),
        )
    if axis == "z":
        multiplier = 1 / min(spacing[0], spacing[1])
        dsize = (
            int(multiplier * spacing[0] * shape[0]),
            int(multiplier * spacing[1] * shape[1]),
        )
    return dsize


def create_dataset(project_id, name):
    src_ds = g.datasets[g.volume_parent[g.volumes[g.selected_volume_idx].id]]
    description = f'This dataset is created by "Slice volume" application from {src_ds.name} dataset (id: {src_ds.id})'
    new_ds_info = g.api.dataset.create(
        project_id=project_id,
        name=name,
        description=description,
        change_name_if_conflict=True,
    )
    return new_ds_info


def get_or_create_dataset(config):
    if config["new"] == True:
        new_ds_info = create_dataset(config["project_id"], config["name"])
        dst_ds_id = new_ds_info.id
    else:
        dst_ds_id = config["dataset_id"]
    return dst_ds_id


def get_plane(volume_ann, axis):
    if axis == "x":
        plane = volume_ann.plane_sagittal
    if axis == "y":
        plane = volume_ann.plane_coronal
    if axis == "z":
        plane = volume_ann.plane_axial
    return plane


def update_dst_project_meta(dst_project_id):
    dst_project_meta_json = g.api.project.get_meta(dst_project_id)
    dst_project_meta = sly.ProjectMeta.from_json(dst_project_meta_json)
    for obj_class in g.project_meta.obj_classes:
        geom_type = (
            obj_class.geometry_type
            if obj_class.geometry_type in [sly.Bitmap, sly.AnyGeometry]
            else sly.AnyGeometry
        )
        obj_class = obj_class.clone(geometry_type=geom_type)
        if dst_project_meta.get_obj_class(obj_class.name) is not None:
            dst_project_meta = dst_project_meta.delete_obj_class(obj_class.name)
        dst_project_meta = dst_project_meta.add_obj_class(obj_class)

    g.api.project.update_meta(dst_project_id, dst_project_meta)
    return dst_project_meta


def get_frame(data, axis, frame_n):
    if axis == "x":
        img = data[frame_n, :, :]
    if axis == "y":
        img = data[:, frame_n, :]
    if axis == "z":
        img = data[:, :, frame_n]
    return img


def get_image(vol_data, axis, frame_n, is_nrrd):
    img = get_frame(vol_data, axis, frame_n)
    if is_nrrd:
        return img
    img = cv2.normalize(
        img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def transpose_geometry(geometry: sly.AnyGeometry) -> sly.AnyGeometry:
    if type(geometry) is sly.Bitmap:
        data = geometry.data
        data = data.transpose()
        origin = sly.PointLocation(row=geometry.origin.col, col=geometry.origin.row)
        geometry = sly.Bitmap(data, origin)
    elif type(geometry) is sly.Rectangle:
        geometry = sly.Rectangle(
            geometry.left, geometry.top, geometry.right, geometry.bottom
        )
    elif type(geometry) is sly.Polygon:
        exterior = [sly.PointLocation(p.col, p.row) for p in geometry.exterior]
        interior = [sly.PointLocation(p.col, p.row) for p in geometry.interior]
        geometry = sly.Polygon(exterior, interior)

    return geometry


def add_labels(
    img_ann: sly.Annotation,
    volume_ann: sly.VolumeAnnotation,
    spatial_figures_masks,
    axis,
    frame_n,
    dst_project_meta: sly.ProjectMeta,
):
    # add labels from slices
    plane = get_plane(volume_ann, axis)
    slice = plane.get(frame_n)
    if slice is not None:
        for figure in slice.figures:
            figure_obj_class_id = figure.parent_object.class_id
            figure_obj_class_name = g.obj_classes[figure_obj_class_id].name
            figure_obj_class = dst_project_meta.get_obj_class(figure_obj_class_name)
            geometry = figure.geometry
            geometry = transpose_geometry(geometry)
            label_tags = sly.TagCollection()
            for figure_tag in figure.parent_object.tags:
                label_tags = label_tags.add(
                    sly.Tag(
                        meta=dst_project_meta.tag_metas.get(figure_tag.meta.name),
                        value=figure_tag.value,
                    )
                )
            label = sly.Label(
                geometry=geometry, obj_class=figure_obj_class, tags=label_tags
            )
            img_ann = img_ann.add_label(label)
    # add labels from spatial figures
    for sp_figure in volume_ann.spatial_figures:
        sp_figure: sly.VolumeFigure
        figure_id = g.key_id_map.get_figure_id(sp_figure.key())
        figure_obj_class_id = sp_figure.parent_object.class_id
        figure_obj_class_name = g.obj_classes[figure_obj_class_id].name
        figure_obj_class = dst_project_meta.get_obj_class(figure_obj_class_name)
        label_tags = sly.TagCollection()
        for figure_tag in sp_figure.parent_object.tags:
            label_tags = label_tags.add(
                sly.Tag(
                    meta=dst_project_meta.tag_metas.get(figure_tag.meta.name),
                    value=figure_tag.value,
                )
            )
        mask = spatial_figures_masks[figure_id]
        frame = get_frame(mask, axis, frame_n)
        if not np.any(frame):
            continue
        label = sly.Label(
            geometry=sly.Bitmap(frame), obj_class=figure_obj_class, tags=label_tags
        )
        img_ann = img_ann.add_label(label)
    return img_ann


def get_annotation(
    volume_ann: sly.VolumeAnnotation,
    spatial_figures_masks,
    axis,
    frame_n,
    img_size,
    dst_project_meta: sly.ProjectMeta,
) -> sly.Annotation:
    if volume_ann is None:
        return None
    img_tags = sly.TagCollection()
    for vol_tag in volume_ann.tags:
        tag_meta = dst_project_meta.tag_metas.get(vol_tag.meta.name)
        img_tags = img_tags.add(sly.Tag(meta=tag_meta, value=vol_tag.value))
    img_ann = sly.Annotation(img_size=img_size, img_tags=img_tags)
    img_ann = add_labels(
        img_ann,
        volume_ann,
        spatial_figures_masks,
        axis,
        frame_n,
        dst_project_meta,
    )
    return img_ann


def transform_image_and_annotation(image, annotation, target_size, is_nrrd):
    if annotation is None:
        image = np.rot90(np.flipud(image))
        if is_nrrd:
            return image, None
        image = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_LINEAR)
        return image, None
    if image.dtype == "int32":
        image = image.astype("int16")
    image, annotation = sly.aug.flipud(image, annotation)
    image, annotation = sly.aug.rotate(image, annotation, 90)
    if is_nrrd:
        return image, annotation
    image, annotation = sly.aug.resize(image, annotation, target_size[::-1])
    return image, annotation


def get_image_and_annotation(
    vol_data,
    vol_ann,
    spatial_figures_masks,
    axis,
    frame_n,
    target_size,
    is_nrrd,
    dst_project_meta,
):
    image = get_image(vol_data, axis, frame_n, is_nrrd)
    annotation = get_annotation(
        vol_ann,
        spatial_figures_masks,
        axis,
        frame_n,
        image.shape[:2],
        dst_project_meta,
    )
    image, annotation = transform_image_and_annotation(
        image, annotation, target_size, is_nrrd
    )
    return image, annotation


def get_nrrd_header(volume_meta, axis):
    slice_meta = {}
    if axis == "x":
        spacing = volume_meta["spacing"][1:]
        slice_meta["space directions"] = [[spacing[0], 0], [0, spacing[1]]]
    if axis == "y":
        spacing = volume_meta["spacing"][0::2]
        slice_meta["space directions"] = [[spacing[0], 0], [0, spacing[1]]]
    if axis == "z":
        spacing = volume_meta["spacing"][0:2]
        slice_meta["space directions"] = [[spacing[0], 0], [0, spacing[1]]]
    return slice_meta


def save_nrrd(dst_dataset_id, name, img_data, header):
    name = g.api.image.get_free_name(dst_dataset_id, name)
    _filename = f"./temp-nrrd-file.nrrd"
    nrrd.write(_filename, img_data, header=header, index_order="C")
    image_info = g.api.image.upload_path(dst_dataset_id, name, _filename)
    os.remove(_filename)
    return image_info


def save_image(dst_dataset_id, name, img):
    name = g.api.image.get_free_name(dst_dataset_id, name)
    image_info = g.api.image.upload_np(dst_dataset_id, name, img)
    return image_info


def save_annotation(image_id, annotation):
    if annotation is None:
        return
    g.api.annotation.upload_ann(image_id, annotation)


def slice_volume(config: dict):
    volume = g.volumes[g.selected_volume_idx]
    volume_data, volume_meta = read_volume(volume)
    axis = config["axis"]
    from_frame = max(1, config["from"])
    to_frame = min(get_max_frame_n(volume_data, axis), config["to"])
    step = config["step"]
    ext = config["format"]
    is_nrrd = ext == ".nrrd"
    add_annotations = config["add_ann"]
    target_size = get_target_size(volume_meta["spacing"], volume_data.shape, axis)
    dst_dataset_id = get_or_create_dataset(config)
    dst_dataset_info = g.api.dataset.get_info_by_id(dst_dataset_id)
    dst_project_id = dst_dataset_info.project_id
    vol_ann = None
    dst_project_meta = None
    spatial_figures_masks = None
    if add_annotations:
        g.key_id_map = sly.KeyIdMap()
        dst_project_meta = update_dst_project_meta(dst_project_id)
        vol_ann = get_volume_annotation(volume.id)
        spatial_figures_masks = get_spatial_masks(
            vol_ann, volume_meta, volume_data.shape
        )

    for frame_n in range(from_frame - 1, to_frame, step):
        image, ann = get_image_and_annotation(
            vol_data=volume_data,
            vol_ann=vol_ann,
            spatial_figures_masks=spatial_figures_masks,
            axis=axis,
            frame_n=frame_n,
            target_size=target_size,
            is_nrrd=is_nrrd,
            dst_project_meta=dst_project_meta,
        )
        name = f"{volume.name}-{g.AXIS_NAME[axis]}-{frame_n+1}{ext}"
        if is_nrrd:
            header = get_nrrd_header(volume_meta, axis)
            image_info = save_nrrd(dst_dataset_id, name, image, header)
        else:
            image_info = save_image(dst_dataset_id, name, image)
        if add_annotations:
            save_annotation(image_info.id, ann)
        yield image_info.id
