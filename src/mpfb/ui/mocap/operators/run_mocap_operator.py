# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         23.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  operators for running the motion capture worker
# ------------------------------------------------------------------------------

import os
import bpy

from .... import ClassManager
from ....services.logservice import LogService
from ....services.systemservice import SystemService
from ....services.mocapenvservice import MocapEnvService
from ....services.mocaprunservice import MocapRunService, RunJob
from ...mpfboperator import MpfbOperator
from ..mocapproperties import read_settings
from .install_env_operator import _redraw_ui

_LOG = LogService.get_logger("mocap.run_mocap_operator")

_REDRAW_INTERVAL = 0.5


def _poll_run_job():
    """Timer callback while the worker runs. When its done, the results are loaded into blender."""
    _redraw_ui()
    job = MocapRunService.get_job()
    if job.is_running:
        return _REDRAW_INTERVAL
    MocapRunService.load_results(job)
    _redraw_ui()
    _LOG.info("Worker run finished with status", job.status)
    return None


class MPFB_OT_Mocap_Run_Operator(MpfbOperator):
    """Send the selected video / image and audio to the motion capture models. Runs in the background,
    the raw outputs (summary text, plots, annotated video, csv and npz files) are written into new folder in MPFB2 cache"""
    bl_idname = "mpfb.mocap_run"
    bl_label = "Run Mocap"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        if MocapEnvService.get_job().is_running:
            cls.poll_message_set("The environment install is running")
            return False
        if MocapRunService.get_job().is_running:
            cls.poll_message_set("Motion capture is already running")
            return False
        if not MocapEnvService.is_env_installed():
            cls.poll_message_set("Install the motion capture environment first")
            return False
        return True

    def hardened_execute(self, context):
        settings = read_settings(context.scene)
        problems = MocapRunService.validate_settings(settings)
        if problems:
            self.report({'ERROR'}, problems[0])
            return {'CANCELLED'}
        missing = MocapEnvService.get_missing_models()
        if missing:
            self.report({'ERROR'}, "Models are missing in data/ai_models: " + ", ".join(missing))
            return {'CANCELLED'}

        run_dir = MocapRunService.new_run_dir()
        job_spec = MocapRunService.build_job(settings, run_dir, MocapEnvService.get_ai_models_dir())
        _LOG.info("Starting the motion capture worker", run_dir)
        MocapRunService.start_run(job_spec)
        if not bpy.app.timers.is_registered(_poll_run_job):
            bpy.app.timers.register(_poll_run_job, first_interval=_REDRAW_INTERVAL)
        self.report({'INFO'}, "Motion capture started, results go to " + run_dir)
        return {'FINISHED'}


class MPFB_OT_Mocap_Cancel_Run_Operator(MpfbOperator):
    """Stop the running motion capture worker"""
    bl_idname = "mpfb.mocap_cancel_run"
    bl_label = "Cancel"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        return MocapRunService.get_job().is_running

    def hardened_execute(self, context):
        MocapRunService.cancel_run()
        self.report({'INFO'}, "Cancelling motion capture")
        return {'FINISHED'}


class MPFB_OT_Mocap_Open_Output_Operator(MpfbOperator):
    """Open the folder with the outputs of last motion capture run."""
    bl_idname = "mpfb.mocap_open_output"
    bl_label = "Open output folder"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        job = MocapRunService.get_job()
        return job.status != RunJob.IDLE and os.path.isdir(job.run_dir)

    def hardened_execute(self, context):
        SystemService.open_file_browser(MocapRunService.get_job().run_dir)
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Mocap_Run_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Cancel_Run_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Open_Output_Operator)
