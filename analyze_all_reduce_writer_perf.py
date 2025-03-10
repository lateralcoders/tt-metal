import pandas as pd
import numpy as np
from typing import List, Dict

# Set non-interactive backend before importing matplotlib
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def parse_profile_log(csv_path: str) -> Dict[str, List[int]]:
    """Parse the profile log and extract durations for each zone."""
    df = pd.read_csv(csv_path)

    # Initialize dictionary to store durations and timestamps
    durations = {
        "WaitSem": defaultdict(lambda: defaultdict(list)),
        "MainLoop": defaultdict(lambda: defaultdict(list)),
        "WriteSems": defaultdict(lambda: defaultdict(list)),
        "ConnClose": defaultdict(lambda: defaultdict(list)),
        "BRISC-KERNEL": defaultdict(lambda: defaultdict(list)),
    }
    waitsem_timestamps = defaultdict(lambda: defaultdict(list))  # To store timestamps for WaitSem
    # Store time between runs
    time_between_runs = defaultdict(lambda: defaultdict(list))
    last_end_time = None

    # Group by run ID and zone name
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    grouped = df.groupby(["PCIe_slot", "core_x", "core_y", "run_ID", "zone_name"])

    for (PCIe_slot, core_x, core_y, run_id, zone_name), group in grouped:
        if PCIe_slot != 13:
            continue
        if zone_name in durations:
            start_time = group[group["type"] == "ZONE_START"]["time[cycles_since_reset]"].values[0]
            end_time = group[group["type"] == "ZONE_END"]["time[cycles_since_reset]"].values[0]
            duration = end_time - start_time
            durations[zone_name][core_x][core_y].append(duration)

            # Calculate time between runs for BRISC-KERNEL
            # if zone_name == 'BRISC-KERNEL':
            #     if run_id != 0:
            #         time_between_runs[core_x][core_y].append(start_time - last_end_time)
            #     last_end_time = end_time

            # Store timestamp for WaitSem
            if zone_name == "WaitSem":
                waitsem_timestamps[core_x][core_y].append(duration)

    return durations, waitsem_timestamps, time_between_runs


def analyze_durations(
    durations: Dict[str, Dict[int, Dict[int, List[int]]]],
    waitsem_timestamps: Dict[int, Dict[int, List[int]]],
    time_between_runs: Dict[int, Dict[int, List[int]]],
):
    plt.style.use("seaborn")

    # Get unique core coordinates
    core_coords = set()
    for x in durations["WaitSem"].keys():
        for y in durations["WaitSem"][x].keys():
            core_coords.add((x, y))

    for core_x, core_y in core_coords:
        fig = plt.figure(figsize=(20, 16))
        gs = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])

        axes_map = {
            "WaitSem": plt.subplot(gs[0, 0]),
            "MainLoop": plt.subplot(gs[0, 1]),
            "WriteSems": plt.subplot(gs[1, 0]),
            "ConnClose": plt.subplot(gs[1, 1]),
            "BRISC-KERNEL": plt.subplot(gs[2, 0]),
            "TimeBetweenRuns": plt.subplot(gs[2, 1]),
        }

        stats = {}
        # if core_x in time_between_runs and core_y in time_between_runs[core_x]:
        #     time_between_runs_array = np.array(time_between_runs[core_x][core_y])
        #     stats['TimeBetweenRuns'] = {
        #         'min': np.min(time_between_runs_array),
        #         'max': np.max(time_between_runs_array),
        #         'median': np.median(time_between_runs_array),
        #         'mean': np.mean(time_between_runs_array),
        #         'std': np.std(time_between_runs_array)
        #     }
        #     sns.histplot(time_between_runs_array, ax=axes_map['TimeBetweenRuns'], bins='auto', kde=True)
        #     axes_map['TimeBetweenRuns'].set_title(f'TimeBetweenRuns Distribution (core {core_x},{core_y})')
        #     axes_map['TimeBetweenRuns'].set_xlabel('Cycles')
        #     axes_map['TimeBetweenRuns'].set_ylabel('Count')
        print(f"Core {core_x},{core_y} has {len(durations['WaitSem'][core_x][core_y])} WaitSem durations")
        print_core = len(durations["WaitSem"][core_x][core_y]) > 0
        if not print_core:
            return

        for zone_name, values in durations.items():
            if len(values[core_x][core_y]) > 0:
                values_array = np.array(values[core_x][core_y])

                stats[zone_name] = {
                    "min": np.min(values_array),
                    "max": np.max(values_array),
                    "median": np.median(values_array),
                    "mean": np.mean(values_array),
                    "std": np.std(values_array),
                }

                ax = axes_map[zone_name]
                sns.histplot(values_array, ax=ax, bins=20, kde=True)
                ax.set_title(f"{zone_name} Distribution (core {core_x},{core_y})")
                ax.set_xlabel("Cycles")
                ax.set_ylabel("Count")

        if len(waitsem_timestamps[core_x][core_y]) > 0:
            waitsem_array = np.array(waitsem_timestamps[core_x][core_y])
            waitsem_x_axis = np.arange(len(waitsem_array))

            ax_chrono = plt.subplot(gs[3, :])
            ax_chrono.bar(waitsem_x_axis, waitsem_array, width=0.8)
            ax_chrono.set_title(f"WaitSem Duration Over Time (core {core_x},{core_y})", fontsize=12)
            ax_chrono.set_xlabel("Sample #", fontsize=10)
            ax_chrono.set_ylabel("Wait Duration (cycles)", fontsize=10)
            ax_chrono.tick_params(axis="both", which="major", labelsize=10)
            ax_chrono.grid(True, alpha=0.3)

        plt.suptitle(f"Core ({core_x},{core_y})", fontsize=16)
        plt.tight_layout()

        print(f"\nCore ({core_x},{core_y}) Statistics (in cycles):")
        print("-" * 50)
        for zone_name, stat in stats.items():
            print(f"\n{zone_name}:")
            print(f"  Minimum:  {stat['min']:.1f}")
            print(f"  Maximum:  {stat['max']:.1f}")
            print(f"  Median:   {stat['median']:.1f}")
            print(f"  Average:  {stat['mean']:.1f}")
            print(f"  Std Dev:  {stat['std']:.1f}")

        plt.savefig(f"profile_analysis_core_{core_x}_{core_y}.png")
        print(f"\nPlot saved as 'profile_analysis_core_{core_x}_{core_y}.png'")
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze profile log data")
    parser.add_argument("csv_file", help="Path to the profile log CSV file")
    args = parser.parse_args()

    # Delete the first line of the csv file
    # Only delete if PCIe_slot is not at start of line
    deleted_first_line = False
    with open(args.csv_file, "r") as file:
        lines = file.readlines()
        if not lines[0].startswith("PCIe slot"):  # Only remove if it's not data
            deleted_first_line = True
            lines = lines[1:]

    if deleted_first_line:
        with open(args.csv_file, "w") as file:
            file.writelines(lines)

    durations, waitsem_timestamps, time_between_runs = parse_profile_log(args.csv_file)
    analyze_durations(durations, waitsem_timestamps, time_between_runs)


if __name__ == "__main__":
    main()
