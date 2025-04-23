"""Uses Mujoco to convert from URDF to MJCF files."""

import os
import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union

import mujoco

from urdf2mjcf.utils import iter_meshes, save_xml


def add_compiler(root: ET.Element) -> None:
    element = ET.Element(
        "compiler",
        attrib={
            "angle": "radian",
            "meshdir": "meshes",
            "eulerseq": "zyx",
            "autolimits": "true",
        },
    )

    if isinstance(existing_element := root.find("compiler"), ET.Element):
        root.remove(existing_element)
    root.insert(0, element)


def add_default(root: ET.Element) -> None:
    default = ET.Element("default")

    # Adds default joint options.
    ET.SubElement(
        default,
        "joint",
        attrib={
            "limited": "true",
            "damping": "0.01",
            "armature": "0.01",
            "frictionloss": "0.01",
        },
    )

    # Adds default geom options.
    ET.SubElement(
        default,
        "geom",
        attrib={
            "condim": "4",
            "contype": "1",
            "conaffinity": "15",
            "friction": "0.9 0.2 0.2",
            "solref": "0.001 2",
        },
    )

    # Adds default motor options.
    ET.SubElement(
        default,
        "motor",
        attrib={"ctrllimited": "true"},
    )

    # Adds default equality options.
    ET.SubElement(
        default,
        "equality",
        attrib={"solref": "0.001 2"},
    )

    # Adds default visualgeom options.
    default_element = ET.SubElement(
        default,
        "default",
        attrib={"class": "visualgeom"},
    )
    ET.SubElement(
        default_element,
        "geom",
        attrib={"material": "visualgeom", "condim": "1", "contype": "0", "conaffinity": "0"},
    )

    if isinstance(existing_element := root.find("default"), ET.Element):
        root.remove(existing_element)
    root.insert(0, default)


def add_option(root: ET.Element) -> None:
    element = ET.Element(
        "option",
        attrib={
            "iterations": "50",
            "timestep": "0.001",
            "solver": "PGS",
            "gravity": "0 0 -9.81",
        },
    )

    if isinstance(existing_element := root.find("option"), ET.Element):
        root.remove(existing_element)
    root.insert(0, element)


def add_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add textures and materials
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "name": "texplane",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".0 .0 .0",
            "rgb2": ".8 .8 .8",
            "width": "100",
            "height": "100",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "matplane",
            "reflectance": "0.",
            "texture": "texplane",
            "texrepeat": "1 1",
            "texuniform": "true",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "visualgeom",
            "rgba": "0.5 0.9 0.2 1",
        },
    )


def get_max_foot_distance(root: ET.Element) -> float:
    def recursive_search(element: ET.Element, current_z: float = 0) -> float:
        max_distance = 0.0
        for child in element:
            if child.tag == "body":
                body_pos = child.get("pos")
                if body_pos:
                    body_z = float(body_pos.split()[2])
                else:
                    body_z = 0
                max_distance = max(max_distance, recursive_search(child, current_z + body_z))
            elif child.tag == "geom":
                geom_pos = child.get("pos")
                if geom_pos:
                    geom_z = float(geom_pos.split()[2])
                    max_distance = max(max_distance, -(current_z + geom_z))
        return max_distance

    worldbody = root.find("worldbody")
    if worldbody is None:
        return 0.0
    return recursive_search(worldbody)


def add_root_body(root: ET.Element, fix_base_link=False, imu_site_name='imu') -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Calculate the initial height
    foot_distance = get_max_foot_distance(root)
    epsilon = 0.5 if fix_base_link else 0.01
    initial_height = foot_distance + epsilon

    # Create a root body
    root_body = ET.Element(
        "body",
        attrib={
            "name": "root",
            "pos": f"0 0 {initial_height}",  # Set the initial height
            "quat": "1 0 0 0",
        },
    )

    # Add a freejoint
    if not fix_base_link:
        ET.SubElement(
            root_body,
            "freejoint",
            attrib={"name": "root"},
        )

    # Add imu site
    ET.SubElement(
        root_body,
        "site",
        attrib={
            "name": imu_site_name,
            "size": "0.01",
            "pos": "0 0 0",
        },
    )

    # Move existing bodies and geoms under base_body
    elements_to_move = list(worldbody)
    for elem in elements_to_move:
        if elem.tag in {"body", "geom"}:
            worldbody.remove(elem)
            root_body.append(elem)
    worldbody.append(root_body)


def add_worldbody_elements(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Add ground plane
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            attrib={
                "name": "ground",
                "type": "plane",
                "pos": "0 0 0",  # Changed to "0 0 0"
                "size": "100 100 0.001",
                "quat": "1 0 0 0",
                "material": "matplane",
                "condim": "3",
                "conaffinity": "15",
            },
        ),
    )

    # Add lights
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.6 0.6 0.6",
                "specular": "0.2 0.2 0.2",
                "pos": "0 0 4",
                "dir": "0 0 -1",
            },
        ),
    )
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.4 0.4 0.4",
                "specular": "0.1 0.1 0.1",
                "pos": "0 0 5.0",
                "dir": "0 0 -1",
                "castshadow": "false",
            },
        ),
    )


def add_actuators(root: ET.Element, no_frc_limit: bool = False, actuator_type='motor') -> None:
    actuator_element = ET.Element("actuator")

    # For each joint, add a motor actuator
    for joint in root.iter("joint"):
        joint_name = joint.attrib.get("name")
        if joint_name is None:
            continue
        joint_range = joint.attrib.get("range")
        # print(joint_name, joint_range)
        if joint_range is None:
            joint.attrib['range'] = "-2147483648 +2147483648"

        # Get joint limits if present
        limit_element = joint.find("limit")
        lower_limit = limit_element.get("lower") if limit_element is not None else None
        upper_limit = limit_element.get("upper") if limit_element is not None else None

        if actuator_type == 'motor':
            if no_frc_limit:
                ctrlrange = "-10000 10000"
            elif lower_limit is not None and upper_limit is not None:
                ctrlrange = f"{lower_limit} {upper_limit}"
            else:
                actuatorfrcrange = joint.attrib.get("actuatorfrcrange")
                ctrlrange = actuatorfrcrange if actuatorfrcrange is not None else "-1 1"

            ET.SubElement(
                actuator_element,
                "motor",
                attrib={
                    "name": joint_name,
                    "joint": joint_name,
                    "ctrllimited": "true",
                    "ctrlrange": ctrlrange,
                    "gear": "1",
                },
            )
        elif actuator_type == 'position':
            ctrlrange = joint_range

            ET.SubElement(
                actuator_element,
                "position",
                attrib={
                    "name": joint_name,
                    "joint": joint_name,
                    "ctrllimited": "true",
                    "ctrlrange": ctrlrange,
                },
            )

    if isinstance(existing_element := root.find("actuator"), ET.Element):
        root.remove(existing_element)
    root.append(actuator_element)


def add_sensors(root: ET.Element, imu_site_name = 'imu') -> None:
    sensor_element = ET.Element("sensor")

    # For each actuator, add sensors
    actuators = root.find("actuator")
    if actuators is not None:
        for actuator in actuators.iter("motor"):
            actuator_name = actuator.attrib.get("name")
            if actuator_name is None:
                continue

            # Add actuatorpos sensor
            ET.SubElement(
                sensor_element,
                "actuatorpos",
                attrib={
                    "name": f"{actuator_name}_p",
                    "actuator": actuator_name,
                },
            )

            # Add actuatorvel sensor
            ET.SubElement(
                sensor_element,
                "actuatorvel",
                attrib={
                    "name": f"{actuator_name}_v",
                    "actuator": actuator_name,
                },
            )

            # Add actuatorfrc sensor
            ET.SubElement(
                sensor_element,
                "actuatorfrc",
                attrib={
                    "name": f"{actuator_name}_f",
                    "actuator": actuator_name,
                    "noise": "0.001",
                },
            )

    # Add additional sensors
    imu_site = None
    for site in root.iter("site"):
        if site.attrib.get("name") == imu_site_name:
            imu_site = site
            break

    if imu_site is not None:
        # Add framequat sensor
        ET.SubElement(
            sensor_element,
            "framequat",
            attrib={
                "name": "orientation",
                "objtype": "site",
                "objname": imu_site_name,
            },
        )

        # Add gyro sensor
        ET.SubElement(
            sensor_element,
            "gyro",
            attrib={
                "name": "angular-velocity",
                "site": imu_site_name,
                "cutoff": "34.9",
            },
        )

        # Add gyro sensor
        ET.SubElement(
            sensor_element,
            "accelerometer",
            attrib={
                "name": "accelerometer",
                "site": imu_site_name,
            },
        )

        # Add gyro sensor
        ET.SubElement(
            sensor_element,
            "velocimeter",
            attrib={
                "name": "velocimeter",
                "site": imu_site_name,
            },
        )

    if isinstance(existing_element := root.find("sensor"), ET.Element):
        root.remove(existing_element)
    root.append(sensor_element)


def add_cameras(root: ET.Element, distance: float = 3.0, height_offset: float = 0.5) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    foot_distance = get_max_foot_distance(root)
    camera_height = foot_distance + height_offset

    # Add a fixed camera
    ET.SubElement(
        worldbody,
        "camera",
        attrib={
            "name": "fixed",
            "pos": f"0 {-distance} {camera_height}",
            "xyaxes": "1 0 0 0 0 1",
        },
    )

    # Add a tracking camera
    ET.SubElement(
        worldbody,
        "camera",
        attrib={
            "name": "track",
            "mode": "trackcom",
            "pos": f"0 {-distance} {camera_height}",
            "xyaxes": "1 0 0 0 0 1",
        },
    )


def add_visual_geom_logic(root: ET.Element) -> None:
    """Add visual geom logic to the root element.

    Args:
        root: The root element of the MJCF file.
    """
    for body in root.findall(".//body"):
        original_geoms = list(body.findall("geom"))
        for geom in original_geoms:
            geom.set("class", "visualgeom")
            # Create a new geom element
            new_geom = ET.Element("geom")
            new_geom.set("type", geom.get("type", "sphere"))
            new_geom.set("rgba", geom.get("rgba") or "")

            # Check if geom has mesh or is a box
            if geom.get("mesh") is None:
                new_geom.set("size", geom.get("size") or "")
            else:
                new_geom.set("mesh", geom.get("mesh") or "")
            if geom.get("pos"):
                new_geom.set("pos", geom.get("pos") or "")
            if geom.get("quat"):
                new_geom.set("quat", geom.get("quat") or "")
            new_geom.set("contype", str(0))

            # Append the new geom to the body
            index = list(body).index(geom)
            body.insert(index + 1, new_geom)


def calculate_dof_from_xml(root: ET.Element) -> int:
    """Calculate the total DOF (qpos size) of a robot based on the XML tree.

    Free joints add 7 DOFs, and joints with a 'range' attribute add 1 DOF each.

    Args:
        root: Root element of the MJCF XML.

    Returns:
        Total DOF (qpos size).
    """
    dof = 0

    # Count free joints
    for _ in root.iter("freejoint"):
        dof += 7

    # Count joints with a 'range' attribute
    for joint in root.iter("joint"):
        if "range" in joint.attrib:
            dof += 1

    return dof


def add_default_position(root: ET.Element, default_position: List[float]) -> None:
    """Add a keyframe to the root element with the default start position.

    Args:
        root: The root element of the MJCF file.
        default_position: The default positions of the robot.
    """
    num_dof = calculate_dof_from_xml(root)
    if len(default_position) != num_dof:
        raise ValueError(f"Default position must have {num_dof} values, got {len(default_position)}.")

    # Add the keyframe with the default position
    keyframe = ET.Element("keyframe")
    key = ET.SubElement(keyframe, "key")
    key.set("name", "default")
    key.set("qpos", " ".join(map(str, default_position)))
    root.append(keyframe)


def convert_urdf_to_mjcf(
    urdf_path: Union[str, Path],
    mjcf_path: Union[str, Path, None] = None,
    no_collision_mesh: bool = False,
    copy_meshes: bool = False,
    camera_distance: float = 3.0,
    camera_height_offset: float = 0.5,
    no_frc_limit: bool = False,
    default_position: Union[List[float], None] = None,
    fix_base_link: bool = False,
    cylinder2box: bool = False,
    use_sensor: bool = True,
    actuator_type: str = 'motor',
    verbose: bool = True,
) -> None:
    """Convert a URDF file to an MJCF file.

    Args:
        urdf_path: The path to the URDF file.
        mjcf_path: The path to the MJCF file. If not provided, use the URDF
            path with the extension replaced with ".xml".
        no_collision_mesh: Do not include collision meshes.
        copy_meshes: Copy mesh files to the output MJCF directory if different
            from URDF directory.
        camera_distance: Distance of the fixed camera from the robot.
        camera_height_offset: Height offset of the fixed camera from the robot.
        no_frc_limit: Do not include force limit for the actuators.
        default_position: Default position for the robot.
    """
    urdf_path = Path(urdf_path)
    mjcf_path = Path(mjcf_path) if mjcf_path is not None else urdf_path.with_suffix(".xml")
    if not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Copy URDF file to temp directory
        urdf_dir = urdf_path.parent.resolve()
        temp_urdf_path = temp_dir_path / urdf_path.name
        temp_urdf_path.write_bytes(urdf_path.read_bytes())

        # Copy mesh files to temp directory and potentially to output directory
        mesh_files = []
        for (_, visual_mesh_path), (_, collision_mesh_path) in iter_meshes(urdf_path):
            for mesh_path in list({visual_mesh_path, collision_mesh_path}):
                if mesh_path is not None:
                    rel = Path(os.path.relpath(mesh_path, urdf_dir))
                    temp_mesh_path = temp_dir_path / mesh_path.name
                    if verbose:
                        print(temp_mesh_path, mesh_path, temp_dir_path, rel, temp_mesh_path.parent.resolve())
                    try:
                        os.makedirs(temp_mesh_path.parent, exist_ok=True)
                        temp_mesh_path.symlink_to(mesh_path)
                        if copy_meshes:
                            mesh_files.append(rel)
                    except FileExistsError:
                        pass

        if verbose:
            print('temp_urdf_path', temp_urdf_path)
            os.system(f'find {temp_dir_path.resolve()}')
        urdf_tree = ET.parse(temp_urdf_path)
        for mj in urdf_tree.iter("mujoco"):
            compiler = mj.find('compiler')
            if compiler is None:
                continue
            compiler.attrib.pop('meshdir', None)
        for mesh in urdf_tree.iter("mesh"):
            full_filename = mesh.attrib.get("filename")
            if full_filename is not None:
                mesh.attrib["filename"] = Path(full_filename).name
            if verbose:
                print(full_filename, Path(full_filename).name)
        urdf_tree.write(temp_urdf_path)
        if verbose:
            urdf_tree.write('test.urdf')

        # Load the URDF file with Mujoco and save it as an MJCF file in the temp directory
        temp_mjcf_path = temp_dir_path / mjcf_path.name
        model = mujoco.MjModel.from_xml_path(temp_urdf_path.as_posix())
        mujoco.mj_saveLastXML(temp_mjcf_path.as_posix(), model)

        # Read the MJCF file and update the paths to the meshes
        mjcf_tree = ET.parse(temp_mjcf_path)
        root = mjcf_tree.getroot()

        for asset in mjcf_tree.iter("asset"):
            for mesh in asset.iter("mesh"):
                mesh_name = Path(mesh.attrib["file"]).name
                # Update the file attribute to just the mesh name
                mesh.attrib["file"] = mesh_name

        if no_frc_limit:
            for joint in root.iter("joint"):
                if "actuatorfrcrange" in joint.attrib:
                    del joint.attrib["actuatorfrcrange"]

        # Turn off internal collisions
        if not no_collision_mesh:
            for geom in root.iter("geom"):
                geom.attrib["contype"] = str(1)
                geom.attrib["conaffinity"] = str(0)
                geom.attrib["density"] = str(0)
                geom.attrib["group"] = str(1)

        if cylinder2box:
            for geom in root.iter("geom"):
                geom_type = geom.get("type", "sphere")
                if geom_type != 'cylinder':
                    continue
                new_type = "box"
                geom.set("type", new_type)
                old_size = geom.get("size")
                if old_size:
                    sizes = map(float, old_size.split(' '))
                    r, h = sizes
                    new_sizes = [r, r, h]
                    new_size = ' '.join(map(str, new_sizes))
                    geom.set("size", new_size)
        # Manually set additional options.
        add_default(root)
        add_compiler(root)
        add_option(root)
        add_assets(root)
        add_cameras(root, distance=camera_distance, height_offset=camera_height_offset)
        add_root_body(root, fix_base_link=fix_base_link)
        add_worldbody_elements(root)
        add_actuators(root, no_frc_limit, actuator_type)
        if use_sensor:
            add_sensors(root)
        add_visual_geom_logic(root)
        if default_position is not None:
            add_default_position(root, default_position)

        # Copy mesh files to the output directory.
        if copy_meshes:
            for mesh_file in mesh_files:
                mjcf_mesh_file = mesh_file.name
                mjcf_mesh_path = mjcf_path.parent.resolve() / 'meshes' / mjcf_mesh_file
                mjcf_mesh_path.parent.mkdir(parents=True, exist_ok=True)
                urdf_mesh_path = urdf_dir / mesh_file
                if mjcf_mesh_path.resolve() != urdf_mesh_path.resolve():
                    shutil.copy2(urdf_mesh_path, mjcf_mesh_path)

        # Write the updated MJCF file to the original destination
        save_xml(mjcf_path, mjcf_tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a URDF file to an MJCF file.")
    parser.add_argument("urdf_path", type=str, help="The path to the URDF file.")
    parser.add_argument("--no-collision-mesh", action="store_true", help="Do not include collision meshes.")
    parser.add_argument("--output", type=str, help="The path to the output MJCF file.")
    parser.add_argument("--copy-meshes", action="store_true", help="Copy mesh files to the output MJCF directory.")
    parser.add_argument("--camera-distance", type=float, default=3.0, help="Camera distance from the robot.")
    parser.add_argument("--camera-height-offset", type=float, default=0.5, help="Camera height offset.")
    parser.add_argument("--no-frc-limit", action="store_true", help="Do not include force limit for the actuators.")
    parser.add_argument("--default-position", type=str, help="Default position for the robot.")
    args = parser.parse_args()

    convert_urdf_to_mjcf(
        urdf_path=args.urdf_path,
        mjcf_path=args.output,
        no_collision_mesh=args.no_collision_mesh,
        copy_meshes=args.copy_meshes,
        camera_distance=args.camera_distance,
        camera_height_offset=args.camera_height_offset,
        no_frc_limit=args.no_frc_limit,
        default_position=None if args.default_position is None else list(map(float, args.default_position.split())),
    )


if __name__ == "__main__":
    main()
