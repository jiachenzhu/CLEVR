import os
import json
import random
import argparse

import bpy
from mathutils import Vector
import utils
from add_random_objects import add_random_objects


def main(args):
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_scene_dir, exist_ok=True)

    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    for output_index in range(args.start_index, args.num_images):
        image_path = os.path.join(args.output_image_dir, f"{output_index}.png")
        scene_path = os.path.join(args.output_scene_dir, f"{output_index}.json")

        num_objects = random.randint(args.min_objects, args.max_objects)
        render_scene(
            args,
            num_objects=num_objects,
            output_index=output_index,
            image_path=image_path,
            scene_path=scene_path,
            color_name_to_rgba=color_name_to_rgba,
            material_mapping=material_mapping,
            object_mapping=object_mapping,
            size_mapping=size_mapping,
        )


def render_scene(args, num_objects, output_index, image_path, scene_path, color_name_to_rgba, material_mapping, object_mapping, size_mapping):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later. We use functionality specific to the CYCLES renderer so BLENDER_RENDER cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = image_path
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size

    cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'

    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    bpy.context.scene.cycles.device = 'GPU'

    scene_struct = {
        'image_index': output_index,
        'objects': [],
        'directions': {},
        'relationships': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += 2.0 * args.camera_jitter * (random.random() - 0.5)

    # Figure out the left, up, and behind directions along the plane and record them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += 2.0 * args.key_light_jitter * (random.random() - 0.5)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += 2.0 * args.back_light_jitter * (random.random() - 0.5)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += 2.0 * args.fill_light_jitter * (random.random() - 0.5)

    # Now make some random objects
    objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, color_name_to_rgba, material_mapping, object_mapping, size_mapping)

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    with open(scene_path, 'w') as f:
        json.dump(scene_struct, f)


def compute_all_relationships(scene_struct, eps=0.2):
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


if __name__ == '__main__':
    argv = utils.extract_args()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend')
    parser.add_argument('--properties_json', default='data/properties.json')
    parser.add_argument('--shape_dir', default='data/shapes')
    parser.add_argument('--material_dir', default='data/materials')

    parser.add_argument('--output_image_dir', default='../output/images/')
    parser.add_argument('--output_scene_dir', default='../output/scenes/')

    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--num_images', default=5, type=int)
    parser.add_argument('--width', default=320, type=int)
    parser.add_argument('--height', default=240, type=int)

    parser.add_argument('--camera_jitter', default=0.5, type=float)
    parser.add_argument('--key_light_jitter', default=1.0, type=float)
    parser.add_argument('--fill_light_jitter', default=1.0, type=float)
    parser.add_argument('--back_light_jitter', default=1.0, type=float)

    parser.add_argument('--min_objects', default=3, type=int)
    parser.add_argument('--max_objects', default=10, type=int)
    parser.add_argument('--min_dist', default=0.25, type=float)
    parser.add_argument('--margin', default=0.4, type=float)
    parser.add_argument('--min_pixels_per_object', default=200, type=int)
    parser.add_argument('--max_retries', default=50, type=int)

    parser.add_argument('--render_num_samples', default=512, type=int)
    parser.add_argument('--render_min_bounces', default=8, type=int)
    parser.add_argument('--render_max_bounces', default=8, type=int)
    parser.add_argument('--render_tile_size', default=256, type=int)
    args = parser.parse_args(argv)
    
    main(args)
