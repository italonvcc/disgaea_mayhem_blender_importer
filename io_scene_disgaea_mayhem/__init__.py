"""Disgaea Mayhem model/animation importer for Blender."""

bl_info = {
    "name": "Disgaea Mayhem Importer",
    "author": "italonvcc",
    "version": (0, 3, 0),
    "blender": (4, 5, 0),
    "location": "File > Import; 3D View > Sidebar > Disgaea",
    "description": "Import NMBM characters, textures, shape keys and skeletal/facial animations",
    "category": "Import-Export",
    "doc_url": "https://github.com/italonvcc/disgaea_mayhem_extraction/blob/main/docs/BLENDER_ADDON.md",
}

from pathlib import Path
import traceback

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import workflow


def clip_changed(self, context):
    if 0 <= self.dm_clip_index < len(self.dm_clips):
        action = self.dm_clips[self.dm_clip_index].action
        if action:
            workflow.activate_action(self, action)


class DM_Clip(bpy.types.PropertyGroup):
    action: PointerProperty(type=bpy.types.Action)


class DM_UL_clips(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name, icon="ACTION")


def report_error(operator, error):
    traceback.print_exc()
    operator.report({"ERROR"}, str(error))
    return {"CANCELLED"}


def report_warnings(operator, warnings):
    if warnings:
        operator.report({"WARNING"}, "; ".join(warnings))


class DM_OT_import_model(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.disgaea_mayhem"
    bl_label = "Import Disgaea Mayhem Model"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".nmbm"

    filter_glob: StringProperty(default="*.nmbm;*.nmb", options={"HIDDEN"})
    texture: StringProperty(
        name="Texture", subtype="FILE_PATH",
        description="Optional NLTX/image for all materials; blank uses each material's original diffuse texture")
    auto_texture: BoolProperty(name="Find Material Textures", default=True)
    animation_mode: EnumProperty(
        name="Animations", default="AUTO",
        items=(("AUTO", "Find Animation Folder", "Load clips from a nearby animation folder"),
               ("NONE", "None", "Import the character in its bind pose"),
               ("FILE", "One Clip", "Choose a single NMBA animation"),
               ("FOLDER", "Choose Folder", "Load all NMBA clips in a chosen folder")))
    animation_file: StringProperty(name="Animation Clip", subtype="FILE_PATH")
    animation_dir: StringProperty(name="Animation Folder", subtype="DIR_PATH")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def draw(self, context):
        self.layout.prop(self, "auto_texture")
        self.layout.prop(self, "texture")
        self.layout.prop(self, "animation_mode")
        if self.animation_mode == "FILE":
            self.layout.prop(self, "animation_file")
        elif self.animation_mode == "FOLDER":
            self.layout.prop(self, "animation_dir")

    def execute(self, context):
        context.window_manager.progress_begin(0, 3)
        try:
            source = Path(self.filepath).resolve()
            if not source.is_file():
                raise ValueError("Choose an existing NMBM model file")
            paths, warnings = [], []
            if self.animation_mode == "FILE":
                if not self.animation_file:
                    raise ValueError("Choose an Animation Clip")
                paths = [Path(bpy.path.abspath(self.animation_file))]
            elif self.animation_mode in {"AUTO", "FOLDER"}:
                if self.animation_mode == "FOLDER" and not self.animation_dir:
                    raise ValueError("Choose an Animation Folder")
                directory = (Path(bpy.path.abspath(self.animation_dir))
                             if self.animation_mode == "FOLDER"
                             else workflow.matching_animation_dir(source))
                if directory:
                    paths = workflow.animation_files(directory)
                if not paths:
                    if self.animation_mode == "FOLDER":
                        raise ValueError("The chosen folder contains no NMBA/NMA clips")
                    warnings.append("No animation clips found; imported the bind pose")
            context.window_manager.progress_update(1)
            rig, meshes, actions, messages = workflow.import_character(
                context, source,
                texture=Path(bpy.path.abspath(self.texture)) if self.texture else None,
                auto_texture=self.auto_texture, paths=paths)
            warnings.extend(messages)
            context.window_manager.progress_update(3)
            self.report({"INFO"}, f"Imported {len(meshes)} meshes, {len(rig.data.bones)} bones, {len(actions)} clips")
            report_warnings(self, warnings)
            return {"FINISHED"}
        except Exception as error:
            return report_error(self, error)
        finally:
            context.window_manager.progress_end()


class DM_OT_import_animations(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.disgaea_mayhem_animations"
    bl_label = "Add Disgaea Animations"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".nmba"
    filter_glob: StringProperty(default="*.nmba;*.nma", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: StringProperty(subtype="DIR_PATH")
    rig_name: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT", "POSE"} and workflow.selected_rig(context) is not None

    def invoke(self, context, event):
        rig = workflow.selected_rig(context)
        self.rig_name = rig.name
        directory = workflow.matching_animation_dir(Path(bpy.path.abspath(rig["dm_source"])))
        if directory:
            self.filepath = str(directory) + "/"
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            rig = bpy.data.objects.get(self.rig_name) if self.rig_name else workflow.selected_rig(context)
            if rig is None:
                raise ValueError("Select an imported Disgaea armature")
            paths = ([Path(self.directory) / f.name for f in self.files]
                     if self.files else [Path(self.filepath)])
            actions, warnings = workflow.add_clips(context, rig, paths)
            self.report({"INFO"}, f"Added {len(actions)} clips; already imported clips were skipped")
            report_warnings(self, warnings)
            return {"FINISHED"}
        except Exception as error:
            return report_error(self, error)


class DM_OT_export_fbx(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.disgaea_mayhem"
    bl_label = "Export Disgaea FBX"
    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx", options={"HIDDEN"})
    rig_name: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    clip_mode: EnumProperty(
        name="Animations", default="ALL",
        items=(("ALL", "All Character Clips", "Export this character's imported clips"),
               ("CURRENT", "Current Clip", "Export only the active clip"),
               ("NONE", "Bind Pose Only", "Export the rig and weights without animation")))

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT", "POSE"} and workflow.selected_rig(context) is not None

    def invoke(self, context, event):
        rig = workflow.selected_rig(context)
        self.rig_name = rig.name
        self.filepath = rig.name + ".fbx"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            rig = bpy.data.objects.get(self.rig_name) if self.rig_name else workflow.selected_rig(context)
            if rig is None:
                raise ValueError("Select an imported Disgaea armature")
            destination = Path(bpy.path.ensure_ext(self.filepath, ".fbx"))
            workflow.export_character(context, destination, rig, self.clip_mode)
            self.report({"INFO"}, f"Exported {destination.name}")
            return {"FINISHED"}
        except Exception as error:
            return report_error(self, error)


class DM_OT_select_face(bpy.types.Operator):
    bl_idname = "object.disgaea_mayhem_select_face"
    bl_label = "Select Facial Mesh"
    bl_description = "Select the mesh with expressions; edit them in Object Data Properties > Shape Keys"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        rig = workflow.selected_rig(context)
        return (context.mode in {"OBJECT", "POSE"} and rig is not None
                and any(o.type == "MESH" and o.data.shape_keys for o in rig.children))

    def execute(self, context):
        rig = workflow.selected_rig(context)
        face = next(o for o in rig.children if o.type == "MESH" and o.data.shape_keys)
        if context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for obj in context.selected_objects:
            obj.select_set(False)
        face.select_set(True)
        context.view_layer.objects.active = face
        return {"FINISHED"}


class DM_PT_character(bpy.types.Panel):
    bl_label = "Disgaea Mayhem"
    bl_idname = "DM_PT_character"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Disgaea"

    def draw(self, context):
        layout = self.layout
        layout.operator(DM_OT_import_model.bl_idname, text="Import Model", icon="IMPORT")
        rig = workflow.selected_rig(context)
        if rig is None:
            layout.label(text="Select an imported character")
            return
        layout.label(text=rig.name, icon="ARMATURE_DATA")
        if rig.dm_clips:
            layout.template_list("DM_UL_clips", "", rig, "dm_clips", rig, "dm_clip_index", rows=6)
            layout.operator("screen.animation_play", text="Play / Pause", icon="PLAY")
        layout.operator(DM_OT_import_animations.bl_idname, text="Add Animations", icon="ACTION")
        faces = [o for o in rig.children if o.type == "MESH" and o.data.shape_keys]
        if faces:
            names = {key.name for o in faces for key in o.data.shape_keys.key_blocks[1:]}
            layout.label(text=f"{len(names)} facial expression controls")
            layout.operator(DM_OT_select_face.bl_idname, icon="SHAPEKEY_DATA")
        layout.operator(DM_OT_export_fbx.bl_idname, text="Export FBX", icon="EXPORT")
        layout.label(text="Save .blend with File > Save As")


def menu_import(self, context):
    self.layout.operator(DM_OT_import_model.bl_idname, text="Disgaea Mayhem (.nmbm)")
    self.layout.operator(DM_OT_import_animations.bl_idname, text="Disgaea Animations (.nmba)")


def menu_export(self, context):
    self.layout.operator(DM_OT_export_fbx.bl_idname, text="Disgaea Mayhem (.fbx)")


CLASSES = (DM_Clip, DM_UL_clips, DM_OT_import_model, DM_OT_import_animations,
           DM_OT_export_fbx, DM_OT_select_face, DM_PT_character)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.dm_clips = CollectionProperty(type=DM_Clip)
    bpy.types.Object.dm_clip_index = IntProperty(default=0, min=0, update=clip_changed)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    del bpy.types.Object.dm_clip_index
    del bpy.types.Object.dm_clips
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
