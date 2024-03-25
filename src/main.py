from supervisely import Application as SlyApp
import src.ui as ui


app = SlyApp(layout=ui.layout)
if len(ui.volume_selector_items) > 0:
    ui.select_volume(0)
