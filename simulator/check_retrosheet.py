import glob
import os

RAW_DIR = r"C:\Users\super\baseball-model\simulator\retrosheet_raw"

for year in range(2021, 2026):
    evn_files = glob.glob(os.path.join(RAW_DIR, f"{year}*.EVN"))
    eva_files = glob.glob(os.path.join(RAW_DIR, f"{year}*.EVA"))
    ros_files = glob.glob(os.path.join(RAW_DIR, f"*{year}.ROS"))
    event_count = len(evn_files) + len(eva_files)
    print(f"{year}: {event_count} EVN/EVA event files, {len(ros_files)} ROS roster files")

all_ros = glob.glob(os.path.join(RAW_DIR, "*.ROS"))
if all_ros:
    sample = all_ros[0]
    print(f"\nSample ROS file: {os.path.basename(sample)}")
    with open(sample, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(line.rstrip())
else:
    print("\nNo .ROS files found in RAW_DIR.")
