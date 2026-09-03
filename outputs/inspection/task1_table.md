# Task 1 — characterisation summary

`declared fps` below is the one label that traces to a real field about *this file's* content: the SPS/VUI `time_scale/num_units_in_tick`, independently confirmed against ffprobe's `r_frame_rate` (the two agree exactly). Two other candidate labels were found, investigated and rejected rather than silently omitted — see "Other frame-rate labels" below. `declared fps` is still not the actual rate; see `measured fps`, computed from `timestamps.txt` interval arithmetic. Full provenance for every figure is in the per-camera JSON files next to this table.

| run/cam | frames (decoded / lines) | resolution | file size | declared fps (nominal) | measured fps | route time (s) | recorded (UTC) |
|---|---|---|---|---|---|---|---|
| runA/cam0 | 2674 / 2695 | 1440x1080 | 128,974,848 B | 10.00 | 9.992 | 267.50 | 2025-02-28T06:24:09.579560+00:00 |
| runA/cam5 | 2674 / 2695 | 1440x1080 | 121,896,960 B | 10.00 | 9.992 | 267.50 | 2025-02-28T06:24:09.579560+00:00 |
| runB/cam0 | 2622 / 2663 | 1440x1080 | 69,992,448 B | 10.00 | 9.989 | 262.40 | 2024-04-13T09:12:41.984952+00:00 |
| runB/cam5 | 2615 / 2663 | 1440x1080 | 63,963,136 B | 10.00 | 9.989 | 261.69 | 2024-04-13T09:12:41.984952+00:00 |

## Other frame-rate labels found — investigated and rejected, not used above

| label | value | why it is not reported as the declared fps |
|---|---|---|
| filename token | 20 | unexplained -- no field anywhere in the file confirms this represents fps; listed only because of its position in the filename, not because its meaning is known |
| ffprobe avg_frame_rate | 25/1 | proven generic fallback for headerless elementary streams: a synthetic test file encoded at a known true 17 fps still reports avg_frame_rate=25/1, identical to this file |

## Frame-count discrepancy (timestamp lines minus decoded frames)

| run/cam | lines | decoded | nal picture count | discrepancy |
|---|---|---|---|---|
| runA/cam0 | 2695 | 2674 | 2674 | 21 |
| runA/cam5 | 2695 | 2674 | 2674 | 21 |
| runB/cam0 | 2663 | 2622 | 2622 | 41 |
| runB/cam5 | 2663 | 2615 | 2615 | 48 |
