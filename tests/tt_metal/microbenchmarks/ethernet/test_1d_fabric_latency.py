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


def profile_results(
    line_size,
    latency_measurement_worker_line_index,
    latency_ping_message_size_bytes,
    latency_ping_burst_size,
    latency_ping_burst_count,
    add_upstream_fabric_congestion_writers,
    num_downstream_fabric_congestion_writers,
    congestion_writers_message_size,
    congestion_writers_use_mcast,
):
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

    packets_per_src_chip = latency_ping_burst_size * latency_ping_burst_count

    latency_ns = 99999999

    return latency_ns


def run_latency_test(
    line_size,
    latency_measurement_worker_line_index,
    latency_ping_message_size_bytes,
    latency_ping_burst_size,
    latency_ping_burst_count,
    add_upstream_fabric_congestion_writers,
    num_downstream_fabric_congestion_writers,
    congestion_writers_message_size,
    congestion_writers_use_mcast,
    expected_mean_latency_ns,
):
    logger.warning("removing file profile_log_device.csv")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_1d_fabric_latency \
                {line_size} \
                {latency_measurement_worker_line_index} \
                {latency_ping_message_size_bytes} \
                {latency_ping_burst_size} \
                {latency_ping_burst_count} \
                {int(add_upstream_fabric_congestion_writers)} \
                {num_downstream_fabric_congestion_writers} \
                {congestion_writers_message_size} \
                {int(congestion_writers_use_mcast)}"
    rc = os.system(cmd)
    if rc != 0:
        if os.WEXITSTATUS(rc) == 1:
            pytest.skip("Skipping test because it only works with T3000")
            return
        logger.info("Error in running the test")
        assert False

    latency_ns = profile_results(
        line_size,
        latency_measurement_worker_line_index,
        latency_ping_message_size_bytes,
        latency_ping_burst_size,
        latency_ping_burst_count,
        add_upstream_fabric_congestion_writers,
        num_downstream_fabric_congestion_writers,
        congestion_writers_message_size,
        congestion_writers_use_mcast,
    )
    logger.info("latency_ns: {} ns", latency_ns)
    allowable_delta = min(0.2, expected_mean_latency_ns * 0.1)
    assert expected_mean_latency_ns - allowable_delta <= latency_ns <= expected_mean_latency_ns + allowable_delta


#####################################
##        Multicast Tests
#####################################


# 1D All-to-All Multicast
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize("latency_measurement_worker_line_index", [0])  # ,1,2,3,4,5,6])
@pytest.mark.parametrize("latency_ping_burst_size", [1])
@pytest.mark.parametrize("latency_ping_burst_count", [1])  # 000])
@pytest.mark.parametrize("add_upstream_fabric_congestion_writers", [False])
@pytest.mark.parametrize("num_downstream_fabric_congestion_writers", [0])
@pytest.mark.parametrize("congestion_writers_message_size", [0])
@pytest.mark.parametrize("congestion_writers_use_mcast", [False])
@pytest.mark.parametrize("latency_ping_message_size_bytes,expected_mean_latency_ns", [(0, 1000)])  # , (512, 1300)])
def test_1D_fabric_latency_on_uncongested_fabric(
    line_size,
    latency_measurement_worker_line_index,
    latency_ping_message_size_bytes,
    latency_ping_burst_size,
    latency_ping_burst_count,
    add_upstream_fabric_congestion_writers,
    num_downstream_fabric_congestion_writers,
    congestion_writers_message_size,
    congestion_writers_use_mcast,
    expected_mean_latency_ns,
):
    run_latency_test(
        line_size,
        latency_measurement_worker_line_index,
        latency_ping_message_size_bytes,
        latency_ping_burst_size,
        latency_ping_burst_count,
        add_upstream_fabric_congestion_writers,
        num_downstream_fabric_congestion_writers,
        congestion_writers_message_size,
        congestion_writers_use_mcast,
        expected_mean_latency_ns,
    )
