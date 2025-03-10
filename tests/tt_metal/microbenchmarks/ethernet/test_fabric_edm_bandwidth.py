# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def profile_results(is_unicast, num_mcasts, num_unicasts, line_size, packet_size, num_data_streams_override):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    main_test_body_string = "MAIN-WRITE-UNICAST-ZONE" if is_unicast else "MAIN-WRITE-MCAST-ZONE"
    setup.timerAnalysis = {
        main_test_body_string: {
            "across": "device",
            "type": "session_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": main_test_body_string},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": main_test_body_string},
        },
    }
    devices_data = import_log_run_stats(setup)
    devices = list(devices_data["devices"].keys())

    # MAIN-TEST-BODY
    main_loop_cycles = []
    for device in devices:
        main_loop_cycle = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string][
            "stats"
        ]["Average"]
        main_loop_cycles.append(main_loop_cycle)

    packets_per_src_chip = num_unicasts if is_unicast else num_mcasts
    traffic_streams_through_boundary = line_size / 2 if num_data_streams_override is None else num_data_streams_override
    total_byte_sent = packets_per_src_chip * traffic_streams_through_boundary * packet_size
    bandwidth = total_byte_sent / max(main_loop_cycles)

    return bandwidth


def run_fabric_edm(
    is_unicast,
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    expected_bw,
    num_devices_with_workers_override,
):
    logger.warning("removing file profile_log_device.csv")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_fabric_edm \
                {num_mcasts} \
                {num_unicasts} \
                {num_links} \
                {num_op_invocations} \
                {int(line_sync)} \
                {line_size} \
                {packet_size} \
                {0 if num_devices_with_workers_override is None else num_devices_with_workers_override}"
    rc = os.system(cmd)
    if rc != 0:
        if os.WEXITSTATUS(rc) == 1:
            pytest.skip("Skipping test because it only works with T3000")
            return
        logger.info("Error in running the test")
        assert False

    num_data_streams_override = None
    if num_devices_with_workers_override is not None:
        num_data_streams_override = num_devices_with_workers_override
    bandwidth = profile_results(is_unicast, num_mcasts, num_unicasts, line_size, packet_size, num_data_streams_override)
    logger.info("bandwidth: {} B/c", bandwidth)
    allowable_delta = min(0.2, expected_bw * 0.1)
    assert expected_bw - allowable_delta <= bandwidth <= expected_bw + allowable_delta


#####################################
##        Multicast Tests
#####################################


# 1D All-to-All Multicast
@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_devices_with_workers_override", [None])
@pytest.mark.parametrize("packet_size,expected_bw", [(16, 0.02), (512, 0.8), (1088, 1.7), (2048, 3.24), (4096, 6.7)])
def test_1D_fabric_all_to_all_mcast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    num_devices_with_workers_override,
    packet_size,
    expected_bw,
):
    run_fabric_edm(
        False,
        num_mcasts,
        num_unicasts,
        num_links,
        num_op_invocations,
        line_sync,
        line_size,
        packet_size,
        expected_bw,
        num_devices_with_workers_override,
    )


# 1D All-to-All Multicast
@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_devices_with_workers_override", [1])
@pytest.mark.parametrize("packet_size,expected_bw", [(16, 0.037), (512, 1.2), (1088, 2.5), (2048, 4.67), (4096, 9.25)])
def test_1D_fabric_mcast_unidirectional_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    num_devices_with_workers_override,
    packet_size,
    expected_bw,
):
    run_fabric_edm(
        False,
        num_mcasts,
        num_unicasts,
        num_links,
        num_op_invocations,
        line_sync,
        line_size,
        packet_size,
        expected_bw,
        num_devices_with_workers_override,
    )


#####################################
##        Unicast Tests
#####################################


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("num_devices_with_workers_override", [None])
@pytest.mark.parametrize(
    "expected_bw",
    [8.4],
)
def test_fabric_edm_unicast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    num_devices_with_workers_override,
    expected_bw,
):
    run_fabric_edm(
        True,
        num_mcasts,
        num_unicasts,
        num_links,
        num_op_invocations,
        line_sync,
        line_size,
        packet_size,
        expected_bw,
        num_devices_with_workers_override,
    )
