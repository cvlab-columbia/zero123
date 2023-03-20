# Taken from https://github.com/allenai/ai2thor/blob/main/scripts/ai2thor-xorg
# Starts an x-server to support running Blender on a headless machine with
# dedicated NVIDIA GPUs

#!/usr/bin/env python3
import os
import sys
import time
import platform
import re
import shlex
import subprocess
import argparse
import signal

# Turning off automatic black formatting for this script as it breaks quotes.
# fmt: off
from typing import List

PID_FILE = "/var/run/ai2thor-xorg.pid"
CONFIG_FILE = "/tmp/ai2thor-xorg.conf"

DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1024


def process_alive(pid):
    """
    Use kill(0) to determine if pid is alive
    :param pid: process id
    :rtype: bool
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    return True


def find_devices(excluded_device_ids):
    devices = []
    id_counter = 0
    for r in pci_records():
        if r.get("Vendor", "") == "NVIDIA Corporation" and r["Class"] in [
            "VGA compatible controller",
            "3D controller",
        ]:
            bus_id = "PCI:" + ":".join(
                map(lambda x: str(int(x, 16)), re.split(r"[:\.]", r["Slot"]))
            )

            if id_counter not in excluded_device_ids:
                devices.append(bus_id)

            id_counter += 1

    if not devices:
        print("Error: ai2thor-xorg requires at least one NVIDIA device")
        sys.exit(1)

    return devices

def active_display_bus_ids():
    # this determines whether a monitor is connected to the GPU
    # if one is, the following Option is added for the Screen "UseDisplayDevice" "None"
    command = "nvidia-smi --query-gpu=pci.bus_id,display_active --format=csv,noheader"
    active_bus_ids = set()
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    if result.returncode == 0:
        for line in result.stdout.decode().strip().split("\n"):
            nvidia_bus_id, display_status = re.split(r",\s?", line.strip())
            bus_id = "PCI:" + ":".join(
                map(lambda x: str(int(x, 16)), re.split(r"[:\.]", nvidia_bus_id)[1:])
            )
            if display_status.lower() == "enabled":
                active_bus_ids.add(bus_id)

    return active_bus_ids

def pci_records():
    records = []
    command = shlex.split("lspci -vmm")
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(":")[0]] = value

    return records


def read_pid():
    if os.path.isfile(PID_FILE):
        with open(PID_FILE) as f:
            return int(f.read())
    else:
        return None


def start(display: str, excluded_device_ids: List[int], width: int, height: int):
    pid = read_pid()

    if pid and process_alive(pid):
        print("Error: ai2thor-xorg is already running with pid: %s" % pid)
        sys.exit(1)

    with open(CONFIG_FILE, "w") as f:
        f.write(generate_xorg_conf(excluded_device_ids, width=width, height=height))

    log_file = "/var/log/ai2thor-xorg.%s.log" % display
    error_log_file = "/var/log/ai2thor-xorg-error.%s.log" % display
    command = shlex.split(
        "Xorg -quiet -maxclients 1024 -noreset +extension GLX +extension RANDR +extension RENDER -logfile %s -config %s :%s"
        % (log_file, CONFIG_FILE, display)
    )

    pid = None
    with open(error_log_file, "w") as error_log_f:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error_log_f)
        pid = proc.pid
        try:
            proc.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass

    if pid and process_alive(pid):
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))
    else:
        print("Error: error with command '%s'" % " ".join(command))
        with open(error_log_file, "r") as f:
            print(f.read())


def print_config(excluded_device_ids: List[int], width: int, height: int):
    print(generate_xorg_conf(excluded_device_ids, width=width, height=height))


def stop():
    pid = read_pid()
    if pid and process_alive(pid):
        os.kill(pid, signal.SIGTERM)

        for i in range(10):
            time.sleep(0.2)
            if not process_alive(pid):
                os.unlink(PID_FILE)
                break


def generate_xorg_conf(
        excluded_device_ids: List[int], width: int, height: int
):
    devices = find_devices(excluded_device_ids)
    active_display_devices = active_display_bus_ids()

    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    Option         "Interactive" "False"
    {extra_options}
    SubSection     "Display"
        Depth       24
        Virtual {width} {height}
    EndSubSection
EndSection
"""
    screen_records = []
    for i, bus_id in enumerate(devices):
        extra_options = ""
        if bus_id in active_display_devices:
            # See https://github.com/allenai/ai2thor/pull/990
            # when a monitor is connected, this option must be used otherwise
            # Xorg will fail to start
            extra_options = 'Option         "UseDisplayDevice" "None"'
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i, width=width, height=height, extra_options=extra_options))
        screen_records.append(
            'Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i)
        )

    xorg_conf.append(
        server_layout_section.format(screen_records="\n    ".join(screen_records))
    )

    output = "\n".join(xorg_conf)
    return output


# fmt: on

if __name__ == "__main__":
    if os.geteuid() != 0:
        path = os.path.abspath(__file__)
        print("Executing ai2thor-xorg with sudo")
        args = ["--", path] + sys.argv[1:]
        os.execvp("sudo", args)

    if platform.system() != "Linux":
        print("Error: Can only run ai2thor-xorg on linux")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-device",
        help="exclude a specific GPU device",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--width",
        help="width of the screen to start (should be greater than the maximum"
        f" width of any ai2thor instance you will start) [default: {DEFAULT_WIDTH}]",
        type=int,
        default=DEFAULT_WIDTH,
    )
    parser.add_argument(
        "--height",
        help="height of the screen to start (should be greater than the maximum"
        f" height of any ai2thor instance you will start) [default: {DEFAULT_HEIGHT}]",
        type=int,
        default=DEFAULT_HEIGHT,
    )
    parser.add_argument(
        "command",
        help="command to be executed",
        choices=["start", "stop", "print-config"],
    )
    parser.add_argument(
        "display", help="display to be used", nargs="?", type=int, default=0
    )
    args = parser.parse_args()
    if args.command == "start":
        start(
            display=args.display,
            excluded_device_ids=args.exclude_device,
            height=args.height,
            width=args.width,
        )
    elif args.command == "stop":
        stop()
    elif args.command == "print-config":
        print_config(
            excluded_device_ids=args.exclude_device,
            width=args.width,
            height=args.height,
        )
