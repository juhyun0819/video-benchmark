# Daywise data preparation source export

`data_tools/daywise/scripts` preserves three historical preparation tools. Their local directory constants now read environment variables, with generic `work/` defaults. The algorithms, validation assertions, formats, and original timestamp mapping conventions are preserved. `data_tools/export_manifest.json` records each original and exported source SHA-256 plus the exact configuration edits.

| Script | Purpose |
| --- | --- |
| `build_annotations.py` | Probe prepared clip durations with `ffprobe`, construct concatenation offsets, preserve original annotations, and emit daywise annotation/timeline files and subtitles. |
| `build_qa.py` | Check pinned HD-EPIC source-file hashes, preserve original question/choice/answer records, shift explicit TIME tags into day playback coordinates, and separate unsupported or missing-input cases. |
| `package_bundle.py` | Create readable indexes, checksums, and archives from an already prepared annotation/QA package. |

These scripts are historical seven-day preparation tools, not a general dataset downloader or a runnable example dataset. They retain assumptions about 70 prepared clips, 47,372 annotation instances, nine HD-EPIC sources, 30 official QA files, seven days, square 720-pixel clips, and 25 fps. The packaging script also retains snapshot-specific explanatory prose and archive names. Review those assumptions before adapting the tools to a different collection; do not remove checks simply to accept mismatched inputs.

The source data is intentionally absent. The environment template `data_tools/daywise/config.example.sh` contains synthetic paths only:

| Variable | Expected local input or output |
| --- | --- |
| `VIDEOMEM_WORK_ROOT` | Prior `videomem_timetable.json`, `annotations_adapted_v1/` with mapped original annotation records and provenance, and `qa_work/` with the pinned official QA source tree and files. |
| `VIDEOMEM_PACKAGE_OUTPUT` | A dedicated output directory; annotation preparation writes the timeline consumed by QA preparation and packaging. |
| `VIDEOMEM_MEDIA_ROOT` | Prepared clips and their `manifest.json`; paths inside that manifest must resolve to the local media. |
| `HD_EPIC_ORIGINAL_ROOT` | Authorized original source MP4s for duration probes. |
| `HD_EPIC_CONVERSION_ROOT` | Conversion `manifest` and summary records named `conversion_manifest.json` and `summary.json`. |

After supplying compatible local inputs, run the scripts in order: `build_annotations.py`, `build_qa.py`, then `package_bundle.py`. They use Python's standard library and external `ffprobe`. Output writers can replace existing generated files, and the packaging script creates archives beside the package directory; use a dedicated output location. No original tool was executed during export, and the export was checked by parsing the copied Python source and verifying that the original source bytes remained unchanged.

Timestamp alignment here assumes ordered concatenation at 1x speed without cuts, gaps, or overlaps. The scripts preserve the distinction between arithmetic/structural validation and visual verification: they do not establish final-render alignment or revalidate answers after sparse frame conversion. Original input aliases remain scoped to their specified intervals; supplying an entire day changes the original QA task.

No actual QA, annotations, clip inventories, timeline/media manifests, archives, logs, videos, images, model files, or credentials were copied. Two scripts in the local management directory were excluded because they transfer one particular media file from a private host; they are not part of the annotation or memory pipeline. Dataset licenses and access conditions still apply to inputs acquired separately.
