from supervisely import Application as SlyApp
import src.ui as ui


app = SlyApp(layout=ui.layout)
ui.select_volume()
