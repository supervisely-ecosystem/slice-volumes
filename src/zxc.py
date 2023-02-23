import json
import math
import supervisely as sly
import trimesh
import numpy as np
import src.globals as g
from src.utils import (
    read_volume,
    get_max_frame_n,
    get_target_size,
    get_or_create_dataset,
    merge_project_meta,
    get_image,
    transform_image_and_annotation,
    get_plane,
    transpose_geometry,
    get_nrrd_header,
    save_nrrd,
    save_image,
    save_annotation,
    get_frame,
)


def get_volume_annotation(volume_id):
    vol_ann_json = g.api.volume.annotation.download(volume_id)
    vol_ann = sly.VolumeAnnotation.from_json(vol_ann_json, g.project_meta, g.key_id_map)
    return vol_ann


def download_spatial_figures(volume_ann):
    id_to_paths = {}
    for sp_figure in volume_ann.spatial_figures:
        figure_id = g.key_id_map.get_figure_id(sp_figure.key())
        id_to_paths[figure_id] = f"{g.STORAGE_DIR}/{figure_id}.stl"
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
    trans_4x4[:3, :3] = trans_3x3
    trans_4x4[:3, 3] = space_origin

    return trans_4x4


# def mask_from_stl(stl_path, mask_shape):
def mask_from_stl(mask_shape, voxel_to_world, stl_path):
    print(voxel_to_world)
    # world_to_voxel = np.linalg.inv(voxel_to_world)

    mesh = trimesh.load(stl_path)

    min_vec = [float("inf"), float("inf"), float("inf")]
    max_vec = [float("-inf"), float("-inf"), float("-inf")]

    mesh.apply_scale((-1, -1, 1))  # LPS to RAS
    # mesh.apply_transform(world_to_voxel)

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


def get_spatial_masks(
    volume_ann: sly.VolumeAnnotation, nrrd_matrix, spatial_figures_paths, data_shape
):
    spatial_figures_masks = {}
    for sp_figure in volume_ann.spatial_figures:
        figure_id = g.key_id_map.get_figure_id(sp_figure.key())
        stl_path = spatial_figures_paths[figure_id]
        mask = mask_from_stl(data_shape, stl_path, nrrd_matrix)
        spatial_figures_masks[figure_id] = mask
    return spatial_figures_masks


def add_labels(
    img_ann: sly.Annotation,
    volume_ann: sly.VolumeAnnotation,
    spatial_figures_masks,
    axis,
    frame_n,
    img_size,
    dst_project_meta: sly.ProjectMeta,
):
    # add labels from slices
    plane = get_plane(volume_ann, axis)
    slice = plane.get(frame_n)
    if slice is not None:
        for figure in slice.figures:
            figure_obj_class_id = figure.parent_object.class_id
            figure_obj_class = g.obj_classes[figure_obj_class_id]
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
        figure_obj_class = g.obj_classes[figure_obj_class_id]

        label_tags = sly.TagCollection()
        for figure_tag in sp_figure.parent_object.tags:
            label_tags = label_tags.add(
                sly.Tag(
                    meta=dst_project_meta.tag_metas.get(figure_tag.meta.name),
                    value=figure_tag.value,
                )
            )

        mask = spatial_figures_masks[figure_id]
        print("mask:")
        print(mask)
        print()

        print("not empty" if np.any(mask) else "empty")

        from matplotlib import pyplot as plt

        for i in range(0, 321, 10):
            frame = get_frame(mask, axis, i)
            im = plt.imshow(frame)
            plt.show()

        geometry = get_frame(mask, axis, frame_n)
        print(f"spatial figure {figure_id} geometry:")
        print(geometry)
        print()
        if not np.any(geometry):
            continue
        label = sly.Label(
            geometry=sly.Bitmap(geometry), obj_class=figure_obj_class, tags=label_tags
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
        img_size,
        dst_project_meta,
    )
    return img_ann


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


def slice_volume(config: dict):
    volume = g.volumes[g.selected_volume_idx]
    data, meta = read_volume(volume)

    axis = config["axis"]
    from_frame = max(1, config["from"])
    to_frame = min(get_max_frame_n(data, axis), config["to"])
    step = config["step"]
    ext = config["format"]
    is_nrrd = ext == ".nrrd"
    add_annotations = config["add_ann"]

    target_size = get_target_size(meta["spacing"], data.shape, axis)
    dst_dataset_id = get_or_create_dataset(config)
    vol_ann = None
    dst_project_meta = None
    if add_annotations:
        print("loading annotations")
        dst_project_meta = merge_project_meta(dst_dataset_id)
        vol_ann = get_volume_annotation(volume.id)
        print(meta)
        nrrd_matrix = matrix_from_volume_meta(meta)
        spatial_figures_paths = download_spatial_figures(vol_ann)
        spatial_figures_masks = get_spatial_masks(
            vol_ann, nrrd_matrix, spatial_figures_paths, data.shape
        )
        print("done loading annotations")
        print("sp figs:")
        print(json.dumps(spatial_figures_paths, indent=2))
        print()

    for frame_n in range(from_frame - 1, to_frame, step):
        image, ann = get_image_and_annotation(
            vol_data=data,
            vol_ann=vol_ann,
            spatial_figures_masks=spatial_figures_masks,
            axis=axis,
            frame_n=frame_n,
            target_size=target_size,
            is_nrrd=is_nrrd,
            dst_project_meta=dst_project_meta,
        )
        print("image annotation:")
        print(ann)
        print()
        print("image annotation json:")
        print(json.dumps(ann.to_json(), indent=2))
        print()
        print("image annotation labels:")
        print(ann.labels)
        print()
        name = f"{volume.name}-{g.AXIS_NAME[axis]}-{frame_n+1}{ext}"
        if is_nrrd:
            header = get_nrrd_header(meta, axis)
            image_info = save_nrrd(dst_dataset_id, name, image, header)
        else:
            image_info = save_image(dst_dataset_id, name, image)
        if add_annotations:
            save_annotation(image_info.id, ann)
        yield image_info.id
