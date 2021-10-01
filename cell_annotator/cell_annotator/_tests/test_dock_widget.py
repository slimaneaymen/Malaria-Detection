from bigannotator import napari_experimental_provide_dock_widget
import pytest
# from pytestqt.qtbot import QtBot
import napari
# from PySide2 import QWidget
# from napari.conftest import make_test_viewer

# this is your plugin name declared in your napari.plugins entry point
MY_PLUGIN_NAME = "napari-bigannotator"
# # the name of your widget(s)
MY_WIDGET_NAMES = ["Example Q Widget", "example_magic_widget"]
# Example_Q_Widget,example_magic_widget=napari_experimental_provide_dock_widget()

@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_something_with_viewer(widget_name,make_napari_viewer):
    # widget = QWidget()
    # QtBot.addWidget(Example_Q_Widget.)  # tell qtbot to clean this widget later    
    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name = widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1
