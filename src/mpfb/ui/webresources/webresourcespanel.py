"""This file contains the web resources panel."""

from ... import ClassManager
from ...services import LogService
from ...services import UiService
from ..abstractpanel import Abstract_Panel

_LOG = LogService.get_logger("webresources.webresourcespanel")

class MPFB_PT_Web_Resources_Panel(Abstract_Panel):
    """UI for opening web links."""
    bl_label = "Web resources"
    bl_category = UiService.get_value("DEVELOPERCATEGORY")
    bl_options = {'DEFAULT_CLOSED'}
    bl_parent_id = "MPFB_PT_System_Panel"

    def _url(self, layout, label, url):
        weblink = layout.operator("mpfb.web_resource", text=label)
        weblink.url = url

    def draw(self, context):
        _LOG.enter()
        layout = self.layout

        self._url(layout, "Project homepage", "http://static.makehumancommunity.org/mpfb.html")
        self._url(layout, "Source code", "https://github.com/makehumancommunity/mpfb2")
        self._url(layout, "Documentation", "http://static.makehumancommunity.org/mpfb/docs.html")
        self._url(layout, "Get support", "http://www.makehumancommunity.org/forum/")
        self._url(layout, "Report a bug", "https://github.com/makehumancommunity/mpfb2/issues")
        self._url(layout, "Asset packs", "http://static.makehumancommunity.org/assets/assetpacks.html")


ClassManager.add_class(MPFB_PT_Web_Resources_Panel)

