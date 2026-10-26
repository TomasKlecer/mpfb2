# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         20.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  UI panel for the motion capture module
# ------------------------------------------------------------------------------

from ... import ClassManager
from ...services.logservice import LogService
from ...services.uiservice import UiService
from ...services.mocapenvservice import MocapEnvService
from ...ui.abstractpanel import Abstract_Panel

_LOG = LogService.get_logger("ui.mocappanel")

# keeps long log lines from unnecesary copying (labels are clipped by panel width anyway)
MAX_LINE_LENGTH = 120


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    return "%d:%02d" % (seconds // 60, seconds % 60)


def multiline(layout, lines):
    column = layout.column(align=True)
    column.scale_y = 0.8
    for line in lines:
        column.label(text=line)


class MPFB_PT_Mocap_Panel(Abstract_Panel):
    """Motion capture from video and audio. The work is done in the subpanels Run Models, Create Animation and Environmen."""
    bl_label = "Motion Capture"
    bl_category = UiService.get_value("MOCAPCATEGORY")
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 85

    def draw(self, context):
        _LOG.enter()
        layout = self.layout
        job = MocapEnvService.get_job()
        if job.is_running:
            layout.label(text="Installing the environment: step %d/%d (%s)" % (
                job.step_index, job.step_count, format_seconds(job.elapsed_seconds)), icon="TIME")
            layout.label(text="Progress is in Environment below.")
        elif not MocapEnvService.is_env_installed():
            layout.label(text="The models need their environment.", icon="INFO")
            layout.label(text="Install it in Environment below.")
        elif MocapEnvService.get_missing_models():
            layout.label(text="Some models are missing, see Environment.", icon="ERROR")


ClassManager.add_class(MPFB_PT_Mocap_Panel)
