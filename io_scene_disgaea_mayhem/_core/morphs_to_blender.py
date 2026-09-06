"""Blender shape keys and facial channels in the character's slotted actions."""

from contextlib import contextmanager

import bpy
from mathutils import Vector


def add_shape_keys(obj, submesh, vertex_start, morphs, rotation):
    stop = vertex_start + submesh.vertex_count
    if not any(vertex_start <= i < stop for morph in morphs for i, _ in morph.deltas):
        return
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    coordinates = [component for vertex in basis.data for component in vertex.co]
    defaults = {}
    for morph in morphs:
        key = obj.shape_key_add(name=morph.name, from_mix=False)
        values = coordinates.copy()
        for index, delta in morph.deltas:
            if vertex_start <= index < stop:
                converted = rotation @ Vector(delta)
                start = (index - vertex_start) * 3
                for axis in range(3):
                    values[start + axis] += converted[axis]
        key.data.foreach_set("co", values)
        key.value = morph.default
        defaults[morph.name] = morph.default
    keys = obj.data.shape_keys
    keys["nmb_morph_defaults"] = defaults
    keys["nmb_morph_slot"] = f"Morphs_{submesh.identifier}"


def character_keys(armature):
    return [obj.data.shape_keys for obj in armature.children
            if obj.type == "MESH" and obj.data.shape_keys
            and "nmb_morph_slot" in obj.data.shape_keys]


def facial_slot(action, keys):
    return action.slots.get("KE" + keys["nmb_morph_slot"]) if action else None


def create_facial_channels(action, clip, armature):
    keys_list = character_keys(armature)
    tracks = {track.name: track for track in clip.facial_tracks}
    names = {block.name for keys in keys_list for block in keys.key_blocks[1:]}
    missing = tracks.keys() - names
    if missing:
        raise ValueError("Facial targets are missing; reimport the model with this add-on version: "
                         + ", ".join(sorted(missing)))
    strip = action.layers[0].strips[0]
    frames = [1 + t - clip.times[0] for t in clip.times]
    for keys in keys_list:
        slot = action.slots.new(id_type="KEY", name=keys["nmb_morph_slot"])
        curves = strip.channelbags.new(slot).fcurves
        for block in keys.key_blocks[1:]:
            track = tracks.get(block.name)
            samples = track.values if track else (keys["nmb_morph_defaults"][block.name],)
            points = ([frames[0], samples[0], frames[-1], samples[0]] if len(samples) == 1
                      else [value for frame, sample in zip(frames, samples) for value in (frame, sample)])
            if frames[0] == frames[-1]:
                points = points[:2]
            curve = curves.new(block.path_from_id("value"))
            curve.keyframe_points.add(len(points) // 2)
            curve.keyframe_points.foreach_set("co", points)
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
            curve.update()
    action["nma_facial_tracks"] = len(tracks)


def assign_facial_action(armature, action):
    for keys in character_keys(armature):
        slot = facial_slot(action, keys)
        animation = keys.animation_data_create()
        desired = action if slot else None
        if animation.action != desired:
            animation.action = desired
        if slot:
            if animation.action_slot != slot:
                animation.action_slot = slot
        else:
            for block in keys.key_blocks[1:]:
                block.value = keys["nmb_morph_defaults"][block.name]


@contextmanager
def preserve_facial_state(armature, *, rollback_only=False):
    states = []
    for keys in character_keys(armature):
        ad = keys.animation_data
        if ad and ad.use_tweak_mode:
            raise ValueError("Exit shape-key NLA Tweak Mode before this operation")
        states.append((keys, ad is not None, ad.action if ad else None,
                       ad.action_slot if ad else None, ad.use_nla if ad else True,
                       [block.value for block in keys.key_blocks]))
    failed = True
    try:
        yield
        failed = False
    finally:
        if rollback_only and not failed:
            return
        for keys, had_animation, action, slot, use_nla, values in states:
            if had_animation:
                ad = keys.animation_data_create()
                ad.action = action
                if action:
                    ad.action_slot = slot
                ad.use_nla = use_nla
            else:
                keys.animation_data_clear()
            for block, value in zip(keys.key_blocks, values):
                block.value = value


@contextmanager
def sync_facial_export(armature, *, animated, use_nla):
    """Synchronize Key slots while FBX's baker changes the rig's take.

    Blender's FBX exporter switches Object actions/NLA strips only, but samples
    all shape-key values. A temporary pre-frame handler assigns the matching
    Key slot before evaluation. Nothing is installed in the saved .blend.
    """
    keys_list = character_keys(armature)
    with preserve_facial_state(armature):
        for keys in keys_list:
            keys.animation_data_create().use_nla = False

        def sync(_scene, *_args):
            ad = armature.animation_data
            action = ad.action if ad else None
            if use_nla and ad:
                strips = [strip for track in ad.nla_tracks if not track.mute
                          for strip in track.strips if not strip.mute]
                action = strips[0].action if len(strips) == 1 else None
            assign_facial_action(armature, action)

        if animated and keys_list:
            bpy.app.handlers.frame_change_pre.append(sync)
        elif not animated:
            for keys in keys_list:
                keys.animation_data.action = None
                for block in keys.key_blocks[1:]:
                    block.value = 0
        try:
            yield
        finally:
            if sync in bpy.app.handlers.frame_change_pre:
                bpy.app.handlers.frame_change_pre.remove(sync)
