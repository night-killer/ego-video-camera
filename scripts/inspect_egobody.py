#!/usr/bin/env python3
import sys
from _bootstrap import bootstrap

bootstrap()
from ego_video_camera.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--inspect-data", *sys.argv[1:]]))
