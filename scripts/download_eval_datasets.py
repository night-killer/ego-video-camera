#!/usr/bin/env python3
from _bootstrap import bootstrap

bootstrap()

from ego_video_camera.eval_dataset_download import main


if __name__ == "__main__":
    raise SystemExit(main())
