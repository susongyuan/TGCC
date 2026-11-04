"""Refuse ambiguous interrupted directories and commit only a complete new folder."""
import json
import shutil
from pathlib import Path


def assert_launch_state(output, resume=None):
    output=Path(output)
    sources=[output,output/"resume"]
    if resume is not None:sources.append(Path(resume))
    retired=[str(p) for source in sources if source.exists() for p in source.glob("RETIRED*")]
    if retired:raise ValueError("retired training/checkpoint cannot initialize or resume: "+json.dumps(retired))
    residual=[str(output/name) for name in ("resume.old","best.old","resume.tmp","best.tmp") if (output/name).exists()]
    if residual:raise ValueError("interrupted atomic checkpoint directories retained; explicit diagnosis/recovery required before GPU initialization: "+json.dumps(residual))


def commit_complete_directory(temp, latest, old, *, required_files):
    temp,latest,old=map(Path,(temp,latest,old))
    if old.exists():raise ValueError("ambiguous old checkpoint retained; refuse overwrite: "+str(old))
    missing=[name for name in required_files if not (temp/name).is_file()]
    if missing:raise ValueError("new checkpoint incomplete; current checkpoint retained: "+json.dumps(missing))
    if latest.exists():latest.rename(old)
    try:
        temp.rename(latest)
    except BaseException:
        if old.exists() and not latest.exists():old.rename(latest)
        raise
    # The new directory was complete and post-write free-floor checked by caller.
    # Old data is removed only after the new directory is installed.
    if old.exists():shutil.rmtree(old)
