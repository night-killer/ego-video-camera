#!/usr/bin/env python3
import sys

from _bootstrap import bootstrap

bootstrap()

from ego_video_camera.egobody_demo_download import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
