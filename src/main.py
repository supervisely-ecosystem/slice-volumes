from supervisely import Application as SlyApp
import src.ui as ui


app = SlyApp(layout=ui.layout)
ui.select_volume()

server = app.get_server()


@server.on_event("shutdown")
def print_something():
    print("========")
    print("Shutdown")
    print("========")
