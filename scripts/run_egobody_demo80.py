#!/usr/bin/env python3
from _bootstrap import bootstrap

bootstrap()

from ego_video_camera.egobody_demo80_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
