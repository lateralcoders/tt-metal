# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import time
import torch
import pytest
import torch.nn as nn
from loguru import logger
from pathlib import Path
from models.perf.perf_utils import prep_perf_report
from models.experimental.functional_yolov8s_world.tt import ttnn_yolov8s_world

# from models.experimental.functional_yolov8s_world.reference import yolov8s_world
from models.experimental.functional_yolov8s_world.reference import yolov8s_world_utils
from models.utility_functions import is_wormhole_b0, run_for_wormhole_b0

# from models.experimental.functional_yolov8s_world.demo.demo_utils import attempt_load
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.experimental.functional_yolov8s_world.tt.ttnn_yolov8s_world_utils import (
    create_custom_preprocessor,
)
from models.experimental.functional_yolov8s_world.tt.ttnn_yolov8s_world import (
    ttnn_YOLOWorld,
)
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from ttnn.model_preprocessing import preprocess_model_parameters, ParameterDict, ParameterList


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["projections"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_download(file, repo="ultralytics/assets"):
    tests = Path(__file__).parent.parent
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolov8s-world.pt"
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"
        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
            print(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"
        except Exception as e:
            print(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            print(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                print(f"ERROR: Download failure for {msg}")
            else:
                print(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        print("Loading weights from:", weight_path)
        ckpt = torch.load(weight_path, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


try:
    sys.modules["ultralytics"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8s_world_utils

except KeyError:
    print("models.experimental.functional_yolov8s_world.reference.yolov8s_world_utils not found.")


def get_expected_times(name):
    base = {"yolov8s_world": (181.8, 14.8)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        # "True",
    ],
)
def test_perf(device, use_pretrained_weight):
    disable_persistent_kernel_cache()
    torch_input = torch.randn(1, 3, 640, 640)

    ttnn_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(
        ttnn_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    batch_size = torch_input.shape[0]

    state_dict = None
    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
        torch_model.eval()
        state_dict = torch_model.state_dict()

    torch_model = yolov8s_world_utils.YOLOWorld()
    torch_model.eval()
    print("torch_model", torch_model)
    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.model, custom_preprocessor=create_custom_preprocessor(device)
    )
    print("parameters", parameters)
    # print("parameters",parameters["txt_feats"])
    # parameters=parameters["model"]
    for i in [12, 15, 19, 22]:
        parameters["model"][i]["attn"]["gl"]["weight"] = ttnn.to_device(
            parameters["model"][i]["attn"]["gl"]["weight"], device=device
        )
        parameters["model"][i]["attn"]["gl"]["bias"] = ttnn.to_device(
            parameters["model"][i]["attn"]["gl"]["bias"], device=device
        )
        parameters["model"][i]["attn"]["bias"] = ttnn.to_device(parameters["model"][i]["attn"]["bias"], device=device)

    parameters["model"][16] = move_to_device(parameters["model"][16], device)

    parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)

    ttnn_model = ttnn_YOLOWorld(
        device,
        parameters,
    )

    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output, ttnn_model_output_x = ttnn_model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        for i in range(len(ttnn_model_output_x)):
            ttnn.deallocate(ttnn_model_output_x[i])
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov8s_world")

    prep_perf_report(
        model_name="models/experimental/functional_yolov8s_world",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 35.50],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov9c(batch_size, expected_perf):
    subdir = "ttnn_yolov8s_world"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest models/experimental/functional_yolov8s_world/demo/demo.py::test_demo"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov8s_world{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
