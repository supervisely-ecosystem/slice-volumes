from supervisely import Application as SlyApp
import src.ui as ui
import src.globals as g


app = SlyApp(layout=ui.layout)
ui.volume_selector.set_value(g.volumes[0])
ui.select_volume()
