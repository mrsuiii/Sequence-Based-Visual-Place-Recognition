# Ground-truth labelling protocol

## Rule 1: this is blind labelling

While labelling, **do not look at `outputs/mapping.csv`, `mapping_full.csv`, or any figure that
overlays a DTW path**. The whole point of these anchors is to measure the pipeline's accuracy
against a judgement that never saw its output. `groundtruth.make_sheets` only ever reads the raw
JPEG cache — it has no way to leak a prediction into the sheets, but *you* can still leak one by
peeking at the CSV. Label first, look at the pipeline's answer only after every anchor is done.

## What "same place" means

Within **±0.5 s of runB travel** (±5 frames, since runB runs at ~10 Hz measured — `findings.md`).
Lane offset (left/right lane of the same stretch of road) doesn't matter — the same stretch of
road counts as the same place, even from a slightly different lane position. Direction/heading
does matter — a frame looking the other way down the same road is not the same place for this
protocol's purposes.

## The two-pass workflow

1. **Coarse pass**: for the anchor's runA frame, open its reference tile
   (`gt/sheets/anchor_refs/{runA_frame:05d}.jpg` — both runA cameras side by side, index burned
   in) and browse the shared coarse overview (`gt/sheets/coarse/{j:05d}.jpg`, every 20th runB
   frame, both cameras side by side). Find the coarse tile that looks like the same place. Note
   its runB index — that's the anchor's *coarse pick*.
2. Once every anchor has a coarse pick, the fine sheets get generated (`make_sheets` called again
   with `coarse_picks`), one directory per anchor: `gt/sheets/fine_{runA_frame:05d}/`, covering
   every single runB frame within ±20 of the coarse pick.
3. **Fine pass**: browse the fine sheet, find the frame (or narrow contiguous run of frames, if
   several look equally correct — e.g. the vehicle was stopped) that best matches. Record it.

## What to record, per anchor

One row in `gt/anchors_user.csv`:

| column | meaning |
|---|---|
| `runA_frame` | the anchor's runA index |
| `runB_best` | your best single-frame pick from the fine sheet |
| `lo`, `hi` | if several consecutive frames are equally valid (e.g. the vehicle was stationary in both runs at this point), the full range; otherwise `lo == hi == runB_best` |
| `quality` | `sure` (confident, unambiguous), `unsure` (plausible but you're not certain), or `no_match` (nothing in the fine sheet — or even the coarse overview — looks like the same place) |
| `note` | free text — anything worth remembering (rain, occlusion, a repeated-looking building, why you picked a range) |

Scanning sequentially from the previous anchor's runB position is fine — route order is a real
property of the data (both runs traverse the same route in the same direction), not something the
model told you.

## If nothing matches at all

That's a legitimate answer — record `quality=no_match`, leave `runB_best`/`lo`/`hi` empty, and
say why in `note` (e.g. "runB doesn't seem to cover this part of the route" or "too much rain to
tell"). Don't force a guess just to fill the cell.
