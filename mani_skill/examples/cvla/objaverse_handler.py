import json
import gzip
import shutil
import tarfile
import requests
import itertools
from pathlib import Path
from typing import Union, Dict, Literal, Optional

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien

try:
    import trimesh
    import msgpack
except ImportError:
    print(
        "Trimesh, msgpack, PIL, not installed, please install it using `pip install trimesh msgpack pillow` if you would want to convert Spoc objects to .glb"
    )

from mani_skill.envs.utils import randomization
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.examples.cvla.cvla_paths import SPOC_ROOT_PATH, SPOC_DOWNLOAD_URL
try:
    from mani_skill.examples.cvla.dataset_creation.chatgpt_describer import chatgpt_describer
except ImportError:
    print("ChatGPT describer not loaded")
    chatgpt_describer = None


def get_spoc_download_url(uid):
    return f"https://pub-2619544d52bd4f35927b08d301d2aba0.r2.dev/assets/{uid}.tar"


def get_bounding_box_from_dict(box_dict: Dict) -> np.ndarray:
    return np.array(
        [
            [box_dict["min"]["x"], box_dict["min"]["y"], box_dict["min"]["z"]],
            [box_dict["max"]["x"], box_dict["max"]["y"], box_dict["max"]["z"]],
        ]
    )


def get_bounding_box_from_annotation(annotation: Dict) -> np.ndarray:
    return get_bounding_box_from_dict(
        annotation["thor_metadata"]["assetMetadata"]["boundingBox"]
    )


class SpocDatasetBuilder:

    def __init__(
        self,
        spoc_root_path=SPOC_ROOT_PATH,
        maximum_objects: Optional[int] = None,
        only_downloaded=False,
    ):
        """ """
        self.spoc_root_path = spoc_root_path

        # Only load when needed
        self._spoc_annotations = None
        self._filtered_spoc_annotations = None

        self.maximum_objects = maximum_objects
        self._only_downloaded = only_downloaded

        self._reduced_gpt_descriptions = {}

        # self._spoc_filter = lambda

    @property
    def spoc_annotations_path(self):
        return self.spoc_root_path / "annotations.json"

    @property
    def spoc_models_path(self):
        return self.spoc_root_path / "assets"

    @property
    def spoc_annotations(self) -> Dict:
        if self._spoc_annotations is None:
            print("loading spock annotations, will take a while")
            with open(self.spoc_annotations_path, "rb") as f_obj:
                self._spoc_annotations = json.load(f_obj)
        return self._spoc_annotations

    @property
    def filtered_spoc_annotations(self) -> Dict:
        if self._filtered_spoc_annotations is None:
            print("filtering spock annotations.")
            # TODO Add the filter as a class/function here?

            def filter_func(annotation: Dict):
                if annotation["scale"] > 1.0:
                    return False

                if "assetMetadata" not in annotation["thor_metadata"]:
                    return False

                bbox = get_bounding_box_from_annotation(annotation)
                extents = bbox[1] - bbox[0]

                # Nothing larger than 0.20m
                if np.any(extents > 0.20):
                    return False
                # Nothing smaller 0.01m
                if np.any(extents < 0.01):
                    return False
                # Not too elongated
                if np.max(extents) / np.min(extents) > 5:
                    return False

                # Check if the object is downloaded
                if self._only_downloaded:
                    spoc_tar_path = self.spoc_models_path / f"{annotation['uid']}.tar"
                    if not spoc_tar_path.exists():
                        return False

                return True

            self._filtered_spoc_annotations = {
                key: annotation
                for key, annotation in self.spoc_annotations.items()
                if filter_func(annotation)
            }

            if (
                self.maximum_objects and len(self._filtered_spoc_annotations) > self.maximum_objects
            ):
                print(
                    f"Warning: Limiting the number of Spoc objects to {self.maximum_objects} of {len(self.spoc_annotations)}"
                )
                # isslice preverses the input order
                self._filtered_spoc_annotations = dict(
                    itertools.islice(
                        self._filtered_spoc_annotations.items(), self.maximum_objects
                    )
                )
                assert len(self._filtered_spoc_annotations) == self.maximum_objects

        return self._filtered_spoc_annotations

    def __len__(self):
        return len(self.filtered_spoc_annotations)

    def sample_uuids(self, num_objects: int = 1, with_replacement=False):
        uuids = randomization.choice(
            list(self.filtered_spoc_annotations.keys()),
            num_objects,
            with_replacement=with_replacement,
        )
        assert uuids, "No objects found"
        assert len(uuids) == num_objects, "Not enough objects found"
        # Debug uuids
        # uuids = [
        #     "205b422823a9442ca229fd9c4605ef92",
        #     "0abe4423ded041c9833e64e213ec3f99",
        #     "0afb5c6b3ca84171a7741bf64c29b296",
        #     "005a246f8c304e77b27cf11cd53ff4ed",
        #     "031ba5c3f25947db9114f7ab7f8b9e7a",
        # ][:num_objects]
        return uuids

    def get_bbox(self, obj_uuid):
        annotation = self.spoc_annotations[obj_uuid]
        bbox = get_bounding_box_from_annotation(annotation)
        return bbox

    def get_object_scale(self, obj_uuid):
        # Spoc Objects are already correctly scaled in meters
        # global_scale = 1.0

        # Default small scale for debugging
        # scale = 0.1

        # Scale such that the smallest xy extent should fit in the gripper of panda (0.08m) --> 0.07m with margin
        bbox = self.get_bbox(obj_uuid)
        extents = bbox[1] - bbox[0]
        # We only are interested in the xy extents, but since the objects
        # are in a different frame, we need to take indices for 0, 2
        scale = 0.07 / min(extents[0], extents[2])
        # We do not want to make the objects larger
        scale = min(scale, 1.0)

        return scale

    def get_spoc_converted_glb_path(self, obj_uuid, force_regeneration=False):
        spoc_glb_path = self.spoc_models_path / obj_uuid / f"{obj_uuid}.glb"

        if not spoc_glb_path.exists() or force_regeneration:
            self._convert_spoc_to_glb(obj_uuid, spoc_glb_path)

        assert spoc_glb_path.exists(), f"glb file not found {spoc_glb_path}"
        return spoc_glb_path

    def _download_spoc(self, obj_uuid) -> Union[Path, None]:
        spoc_archive_path = (self.spoc_models_path / obj_uuid).with_suffix(".tar")

        if not (spoc_archive_path.exists() and spoc_archive_path.stat().st_size > 8192):
            # Download the archive
            try:
                response = requests.get(
                    get_spoc_download_url(obj_uuid), stream=True
                )  # Use streaming for large files
                response.raise_for_status()  # Raise an exception for HTTP errors
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error occurred {e} when downloading {obj_uuid}")
                return

            with spoc_archive_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):  # Write in chunks
                    file.write(chunk)

        # Check if the archive was already unpacked
        spoc_obj_dir = self.spoc_models_path / obj_uuid
        if not spoc_obj_dir.exists():
            # Unpack the full-archive
            with tarfile.open(spoc_archive_path, "r") as tar:
                tar.extractall(self.spoc_models_path)

            # Unpack the msgpack
            msgpack_path = spoc_obj_dir / f"{obj_uuid}"
            with gzip.open(msgpack_path.with_suffix(".msgpack.gz"), "rb") as f_in:
                with open(msgpack_path.with_suffix(".msgpack"), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # TODO Unpack annotations?

        return spoc_obj_dir

    def get_trimesh_from_spoc(self, obj_uuid, warn=False):
        spoc_obj_path = Path(self._download_spoc(obj_uuid))

        # TODO Fix me?
        msg_path = spoc_obj_path / f"{obj_uuid}.msgpack"
        with open(msg_path, "rb") as data_file:
            byte_data = data_file.read()
        data_loaded = msgpack.unpackb(byte_data)

        # Some debug information
        # for key in data_loaded:
        #     sample = ""
        #     if type(data_loaded[key]) == list:
        #         sample = (len(data_loaded[key]), data_loaded[key][0])
        #     if isinstance(data_loaded[key], (bool, str, dict)):
        #         sample = data_loaded[key]
        #     print(key, type(data_loaded[key]), sample)

        vertices_np = np.array(
            [list(vertex.values()) for vertex in data_loaded["vertices"]]
        )
        faces_np = np.array(data_loaded["triangles"]).reshape(-1, 3)
        normals_np = np.array(
            [list(vertex.values()) for vertex in data_loaded["normals"]]
        )
        uvs_np = np.array([list(vertex.values()) for vertex in data_loaded["uvs"]])

        albedo_img = Image.open(spoc_obj_path / data_loaded["albedoTexturePath"])
        emission_img = Image.open(
            spoc_obj_path / data_loaded["emissionTexturePath"]
        )
        metallic_smoothnes_img = Image.open(
            spoc_obj_path / data_loaded["metallicSmoothnessTexturePath"]
        )
        normals_img = Image.open(spoc_obj_path / data_loaded["normalTexturePath"])

        # random_color_visual = trimesh.visual.ColorVisuals(vertex_colors=np.random.rand(vertices_np.shape[0], 4))

        # When albedo is all black
        if warn and np.all(np.asarray(albedo_img) < 0.001):
            print(
                "[WARNING] All black albedo image, flipping albedo and emission, [NH]: Not sure why this happens"
            )
            albedo_img, emission_img = emission_img, albedo_img

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo_img,
            metallicRoughnessTexture=metallic_smoothnes_img,
            normalTexture=normals_img,
            emissiveTexture=emission_img,
        )

        mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            face_normals=normals_np,
            visual=trimesh.visual.texture.TextureVisuals(uv=uvs_np, material=material),
            # visual=random_color_visual,
        )

        return mesh

    def _convert_spoc_to_glb(self, obj_uuid, spoc_glb_path):
        _ = self._download_spoc(obj_uuid)
        trimesh_mesh = self.get_trimesh_from_spoc(obj_uuid)
        trimesh_mesh.export(spoc_glb_path)

    def get_spoc_descriptions(self, obj_uuid):
        spoc_annotation = self.spoc_annotations[obj_uuid]
        descriptions = [
            value for key, value in spoc_annotation.items() if "description" in key
        ]
        return descriptions

    def get_condensed_gpt_description(self, obj_uuid, recreate=False):
        descriptions = self.get_spoc_descriptions(obj_uuid)

        if obj_uuid in self._reduced_gpt_descriptions:
            return self._reduced_gpt_descriptions[obj_uuid]

        # TODO Check if already saved --> load from file
        chatgpt_description_path = (
            self.spoc_models_path / obj_uuid / "chatgpt_description.json"
        )
        if chatgpt_description_path.exists() and not recreate:
            with chatgpt_description_path.open("r") as f_obj:
                reduced_description = json.load(f_obj)
        elif chatgpt_describer is not None:
            # Model dump converts the pydantic base model to a dict
            reduced_description = chatgpt_describer.describe(descriptions).model_dump()
            with chatgpt_description_path.open("w") as f_obj:
                json.dump(reduced_description, f_obj)
        else:
            raise ValueError(chatgpt_describer)

        self._reduced_gpt_descriptions[obj_uuid] = reduced_description
        return reduced_description

    def get_gpt_name(
        self,
        obj_uuid,
        words: Literal[
            "one_word", "two_words", "three_words", "four_words", "five_words"
        ] = "three_words",
    ):
        return self.get_condensed_gpt_description(obj_uuid)[words]

    def get_gpt_household(self, obj_uuid):
        return self.get_condensed_gpt_description(obj_uuid)[
            "is_common_household_object"
        ]

    def get_object_name(self, obj_uuid):
        return self.spoc_annotations[obj_uuid]["category"]


class SpocDatasetBuilderFast(SpocDatasetBuilder):
    def __init__(self, spoc_root_path=SPOC_ROOT_PATH,
                 annotation_file=SPOC_ROOT_PATH / "clip_description2.jsonl",
                 maximum_objects=None):

        # do this to get files, but try not to call anything that requires loading spoc annotations, otherwise it takes long
        super().__init__(spoc_root_path)

        self.clip_annotation_data = {}
        uuids_seen = set()
        with open(annotation_file, "r") as f_obj:
            for line in f_obj:
                entry = json.loads(line)
                if entry["uid"] in uuids_seen:
                    print("removing duplicate")
                    continue
                self.clip_annotation_data[entry["uid"]] = entry
                uuids_seen.add(entry["uid"])

        def filter_func(annotation: Dict, t_best_descr=-3, t_low_poly=-10, t_grayscale=-12):
            v_low_poly = annotation["quality"]["cartoon low-poly model"]
            v_grayscale = annotation["quality"]["grayscale image"]
            if v_low_poly > t_low_poly:
                return False
            if v_grayscale > t_grayscale:
                return False

            short_descr_values = [v for k, v in annotation["description"].items() if len(k.split()) < 4]
            v_best_descr = max(short_descr_values)
            if v_best_descr < t_best_descr:
                return False

            short_descr_text = [k for k, v in annotation["description"].items() if len(k.split()) < 4]
            best_descr = short_descr_text[np.argmax(short_descr_values)]
            blacklist_words = set(("stone", "rock", "figurine", "ring"))
            if set(best_descr.split()).intersection(blacklist_words):
                return False
            return True

        self.clip_annotation_data_filterd = {}
        for annotation in self.clip_annotation_data.values():
            if filter_func(annotation):
                self.clip_annotation_data_filterd[annotation["uid"]] = annotation

                if maximum_objects is not None and len(self.clip_annotation_data_filterd) == maximum_objects:
                    break

    @property
    def spoc_annotations(self) -> Dict:
        raise ValueError("You can commet this error out if you want.")

    def get_object_scale(self, obj_uuid):
        bbox = np.array(self.clip_annotation_data[obj_uuid]["bbox"])
        extents = bbox[1] - bbox[0]
        # We only are interested in the xy extents, but since the objects
        # are in a different frame, we need to take indices for 0, 2
        scale = 0.07 / min(extents[0], extents[2])
        # We do not want to make the objects larger
        scale = min(scale, 1.0)

        return scale

    def get_object_name(self, obj_uuid, max_len=3):
        annotation = self.clip_annotation_data[obj_uuid]
        short_descr_values = [v for k, v in annotation["description"].items() if len(k.split()) <= max_len]
        short_descr_text = [k for k, v in annotation["description"].items() if len(k.split()) <= max_len]
        best_descr = short_descr_text[np.argmax(short_descr_values)]
        return best_descr

    def sample_uuids(self, num_objects: int = 1, with_replacement=False):
        uuids = randomization.choice(
            list(self.clip_annotation_data_filterd.keys()),
            num_objects,
            with_replacement=with_replacement,
        )
        assert uuids, "No objects found"
        assert len(uuids) == num_objects, "Not enough objects found"
        return uuids


# Specific for Maniskill --> TODO: Refactor into somewhere else when there is time
def get_spoc_builder(
    scene: ManiSkillScene,
    uuid,
    add_collision=True,
    add_visual=True,
    spoc_dataset: SpocDatasetBuilder = None
):
    # We need to rotate the object around z to make it upright?
    #obj_q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))
    obj_q = R.from_rotvec(np.deg2rad(90) * np.array([1, 0, 0])).as_quat(scalar_first=True)

    # TODO Use pose_z_rot_angle from the annotations?
    obj_pose = sapien.Pose(q=obj_q)

    builder = scene.create_actor_builder()
    builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])

    scale = spoc_dataset.get_object_scale(uuid)

    # TODO Set these?
    density = 1000
    # physical_material = None

    glb_path = str(spoc_dataset.get_spoc_converted_glb_path(uuid))

    if add_collision:
        collision_file = glb_path
        # builder.add_nonconvex_collision_from_file # This does not support density
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            # material=physical_material,
            density=density,
            pose=obj_pose,
        )

    if add_visual:
        visual_file = glb_path
        builder.add_visual_from_file(
            filename=visual_file, scale=[scale] * 3, pose=obj_pose
        )

    return builder


if __name__ == "__main__":
    SpocDataset = SpocDatasetBuilder(
        spoc_root_path=SPOC_ROOT_PATH,
        # maximum_objects=100,
        # only_downloaded=True,  # For debug purposes useful
    )
    print(f"Loadable {len(SpocDataset)} Spoc Objects")
