from supervisely.app.widgets import (
    Container,
    Card,
    ProjectThumbnail,
    DatasetThumbnail,
    Select,
    Text,
    InputNumber,
    Field,
    Input,
    SelectDataset,
    SelectProject,
    OneOf,
    Button,
    Progress,
    Checkbox,
)
import supervisely as sly
import src.globals as g
import src.utils as utils


def get_input_thumbnail():
    if g.DATASET_ID is None:
        return ProjectThumbnail(g.project_info)
    return DatasetThumbnail(g.project_info, g.datasets[0])


input_thumbnail = get_input_thumbnail()


input_card = Card(
    title="1️⃣ Input",
    description="Input project or dataset",
    content=Container(widgets=[input_thumbnail]),
)


volume_selector_items = [
    Select.Item(str(i), volume.name) for i, volume in enumerate(g.volumes)
]
volume_selector = Select(items=volume_selector_items)
volume_info_id = Text('<p style="text-align:left">Id:</p>')
volume_info_name = Text('<p style="text-align:left">Name:</p>')
volume_info_axial_frames = Text('<p style="text-align:left">Axial frames:</p>')
volume_info_coronal_frames = Text('<p style="text-align:left">Coronal frames:</p>')
volume_info_sagittal_frames = Text('<p style="text-align:left">Sagittal frames:</p>')
result_text = Text()
volume_info = Container(
    widgets=[
        volume_info_id,
        volume_info_name,
        volume_info_axial_frames,
        volume_info_coronal_frames,
        volume_info_sagittal_frames,
    ],
    gap=0,
)

def select_volume(val):
    def html_wrap_text_align(text, align="right"):
        return f'<p style="text-align:right">{text}</p>'

    def html_wrap_div_space_between(text, justify_content="space-between"):
        return f'<div style="display: flex; flex-direction: row; justify-content: {justify_content};">{text}</div>'

    idx = int(val)
    g.selected_volume_idx = idx
    volume = g.volumes[idx]

    result_text.set("", status="text")
    volume_info_id.set(
        html_wrap_div_space_between(
            html_wrap_text_align("Id:") + html_wrap_text_align(volume.id)
        ),
        status="text",
    )
    volume_info_name.set(
        html_wrap_div_space_between(
            html_wrap_text_align("Name:") + html_wrap_text_align(volume.name)
        ),
        status="text",
    )
    volume_info_axial_frames.set(
        html_wrap_div_space_between(
            html_wrap_text_align("Axial frames:")
            + html_wrap_text_align(volume.meta["dimensionsIJK"]["z"])
        ),
        status="text",
    )
    volume_info_coronal_frames.set(
        html_wrap_div_space_between(
            html_wrap_text_align("Coronal frames:")
            + html_wrap_text_align(volume.meta["dimensionsIJK"]["y"])
        ),
        status="text",
    )
    volume_info_sagittal_frames.set(
        html_wrap_div_space_between(
            html_wrap_text_align("Sagittal frames:")
            + html_wrap_text_align(volume.meta["dimensionsIJK"]["x"])
        ),
        status="text",
    )

if len(volume_selector_items) > 0:
    select_volume(0)

@volume_selector.value_changed
def handle_select_volume(val):
    select_volume(val)


volume_input_card = Card(
    title="2️⃣ Input Volume",
    description="Select Volume",
    content=Container(widgets=[volume_selector, volume_info]),
)

slice_config_widgets = {}
axis_selector = Select(
    items=[
        Select.Item("x", g.AXIS_NAME["x"]),
        Select.Item("y", g.AXIS_NAME["y"]),
        Select.Item("z", g.AXIS_NAME["z"]),
    ]
)
slice_config_widgets["axis"] = Field(
    title="Select Axis", description="Select axis", content=axis_selector
)
from_frame_input = InputNumber(min=1)
slice_config_widgets["from"] = Field(
    title="From frame #",
    description="Input starting frame number",
    content=from_frame_input,
)
to_frame_input = InputNumber()
slice_config_widgets["to"] = Field(
    title="To frame #", description="Input ending frame number", content=to_frame_input
)
frame_step_input = InputNumber(min=1)
slice_config_widgets["step"] = Field(
    title="Step", description="Input step", content=frame_step_input
)


# @axis_selector.value_changed
# def axis_changed(val):
#     volume = g.volumes[g.selected_volume_idx]
#     from_frame_input.max = volume.meta["dimensionsIJK"][val]
#     to_frame_input.max = volume.meta["dimensionsIJK"][val]


# @from_frame_input.value_changed
# def from_frame_changed(val):
#     to_frame_input.min = val


# @to_frame_input.value_changed
# def to_frame_changed(val):
#     from_frame_input.max = val


add_annotation_checkbox = Checkbox(content=Text("Add annotations"))
slice_config_widgets["add_ann"] = Field(
    title="Add annotations",
    description="Check if you would like to add annotations from volume",
    content=add_annotation_checkbox,
)

new_dataset_name_input = Input()
new_dataset_select_project = SelectProject(allowed_types=[sly.ProjectType.IMAGES])
new_dataset = Container(
    widgets=[
        new_dataset_select_project,
        Field(title="Dataset Name", content=new_dataset_name_input),
    ]
)
# how to set ProjectType ?
existing_dataset = SelectDataset()
output_selector = Select(
    items=[
        Select.Item("new_dataset", "in new dataset", new_dataset),
        Select.Item("existing_dataset", "in existing dataset", existing_dataset),
    ]
)
select_output = Field(output_selector, title="Where to save images:")
selected_output_config = OneOf(output_selector)
format_selector = Select(
    items=[
        Select.Item(".png", "png"),
        Select.Item(".jpeg", "jpeg"),
        Select.Item(".nrrd", "nrrd"),
        Select.Item(".mpo", "mpo"),
        Select.Item(".bmp", "bmp"),
        Select.Item(".webp", "webp"),
        Select.Item(".tiff", "tiff"),
    ]
)
output_format = Field(
    title="Image format",
    description="Select resulting image format",
    content=format_selector,
)

slice_config_card = Card(
    title="3️⃣ Slice config",
    description="Input slicing config",
    content=Container(widgets=slice_config_widgets.values()),
)
output_config_card = Card(
    title="4️⃣ Output config",
    description="Select output dataset",
    content=Container(widgets=[select_output, selected_output_config, output_format]),
)
config_cards = Container(
    widgets=[volume_input_card, slice_config_card, output_config_card],
    direction="horizontal",
)

start_button = Button(text="start")
upload_progress = Progress()


def get_config():
    config = {"format": format_selector.get_value()}
    for key, field in slice_config_widgets.items():
        content = field._content
        if type(content) is Checkbox:
            config[key] = content.is_checked()
        else:
            config[key] = content.get_value()
    if output_selector.get_value() == "new_dataset":
        config = {
            **config,
            "new": True,
            "project_id": new_dataset_select_project.get_selected_id(),
            "name": new_dataset_name_input.get_value(),
        }
    else:
        config = {
            **config,
            "new": False,
            "dataset_id": existing_dataset.get_selected_id(),
        }
    return config


def config_is_valid(config):
    from_frame = config["from"]
    to_frame = config["to"]
    step = config["step"]
    if from_frame > to_frame:
        return False
    new = config["new"]
    if new:
        if config["project_id"] is None:
            return False
        if config["name"] == "":
            return False
    else:
        if config["dataset_id"] is None:
            return False
    return True


def get_max_frame_n(axis):
    volume = g.volumes[g.selected_volume_idx]
    return volume.meta["dimensionsIJK"][axis]


@start_button.click
def start():
    result_text.set("Slicing in progress...", status="text")
    config = get_config()
    if not config_is_valid(config):
        result_text.set("Invalid slice config. Please update settings and try again", status="error")
        return
    axis = config["axis"]
    from_frame = config["from"]
    to_frame = min(get_max_frame_n(axis), config["to"])
    step = config["step"]
    with upload_progress(total=(to_frame - from_frame) // step + 1) as pbar:
        for _ in utils.slice_volume(config):
            pbar.update(1)
    result_text.set("Slicing completed successfully", status="success")


layout = Container(widgets=[input_card, config_cards, start_button, result_text, upload_progress])
