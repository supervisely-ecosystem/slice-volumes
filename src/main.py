from supervisely import Application as SlyApp
import src.ui as ui
import src.globals as g


app = SlyApp(layout=ui.layout)
ui.select_volume(g.volumes[0])
