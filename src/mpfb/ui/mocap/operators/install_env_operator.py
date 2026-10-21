# ------------------------------------------------------------------------------
# MPFB2 Extension by Klecer
# Author:       Tomáš Klecer
# Date:         20.10.2026
# University:   Brno University of Technology
# Supervisor:   Ing. Tomáš Chlubna, Ph.D.
# Description:  operators for installing the python environment of motion capture models
# ------------------------------------------------------------------------------

import os
import bpy

from .... import ClassManager
from ....services.logservice import LogService
from ....services.systemservice import SystemService
from ....services.mocapenvservice import MocapEnvService, InstallJob
from ...mpfboperator import MpfbOperator

_LOG = LogService.get_logger("mocap.install_env_operator")

_REDRAW_INTERVAL = 0.5


def _redraw_ui():
    """Tag the 3D view sidebars for redraw, so the panel shows the progress of the install."""
    window_manager = bpy.context.window_manager
    if window_manager is None:
        return
    for window in window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'UI':
                        region.tag_redraw()


def _poll_install_job():
    """Timer callback, runs while the install job is running."""
    _redraw_ui()
    job = MocapEnvService.get_job()
    if job.is_running:
        return _REDRAW_INTERVAL
    _LOG.info("Install job finished with status", job.status)
    return None


class MPFB_OT_Mocap_Install_Env_Operator(MpfbOperator):
    """Create the uv managed Python environment for motion capture models
    and install the packages they need (mediapipe, torch, transformers). Runs in the background
    """
    bl_idname = "mpfb.mocap_install_env"
    bl_label = "Install motion capture environment"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        if not bpy.app.online_access:
            cls.poll_message_set("Online access is disabled in Preferences > System > Network")
            return False
        if MocapEnvService.get_job().is_running:
            cls.poll_message_set("An install is already running")
            return False
        return True

    def hardened_execute(self, context):
        env_dir = MocapEnvService.get_env_dir()
        own_uv = MocapEnvService.get_own_uv_binary()

        if MocapEnvService.get_preferences() is not None and MocapEnvService.get_preferences().mocap_use_own_uv:
            if not own_uv or not os.path.isfile(own_uv):
                self.report({'ERROR'}, "Select an existing uv binary, or untick \"Use own uv binary\"")
                return {'CANCELLED'}

        try:
            os.makedirs(env_dir, exist_ok=True)
        except OSError as err:
            self.report({'ERROR'}, "Can not create the environment folder %s: %s" % (env_dir, err))
            return {'CANCELLED'}

        _LOG.info("Starting motion capture environment install", (env_dir, own_uv))
        MocapEnvService.start_install(env_dir, own_uv)
        if not bpy.app.timers.is_registered(_poll_install_job):
            bpy.app.timers.register(_poll_install_job, first_interval=_REDRAW_INTERVAL)
        self.report({'INFO'}, "Installing the motion capture environment into " + env_dir)
        return {'FINISHED'}


class MPFB_OT_Mocap_Cancel_Install_Operator(MpfbOperator):
    """Stop the running install of the motion capture environment"""
    bl_idname = "mpfb.mocap_cancel_install"
    bl_label = "Cancel"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        return MocapEnvService.get_job().is_running

    def hardened_execute(self, context):
        MocapEnvService.cancel_install()
        self.report({'INFO'}, "Cancelling the install")
        return {'FINISHED'}


class MPFB_OT_Mocap_Open_Env_Dir_Operator(MpfbOperator):
    """Open folder of motion capture environment which contains install.log"""
    bl_idname = "mpfb.mocap_open_env_dir"
    bl_label = "Open environment folder"
    bl_options = {'REGISTER'}

    def get_logger(self):
        return _LOG

    def hardened_execute(self, context):
        job = MocapEnvService.get_job()
        env_dir = job.env_dir if job.status != InstallJob.IDLE and job.env_dir else MocapEnvService.get_env_dir()
        if not os.path.isdir(env_dir):
            self.report({'ERROR'}, "The folder does not exist: " + env_dir)
            return {'CANCELLED'}
        SystemService.open_file_browser(env_dir)
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Mocap_Install_Env_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Cancel_Install_Operator)
ClassManager.add_class(MPFB_OT_Mocap_Open_Env_Dir_Operator)
