# Per-cell labels

Drop annotated cells into `labels.csv` in this directory. Coordinate-based —
no need to know instance IDs. The Stage 3 crop builder resolves
`(source_image, x, y)` → `instance_id` via point-in-mask.

## Schema

| column | type | required | description |
|---|---|---|---|
| `source_image` | str | yes | stem matching files in `raw_images/` and `unet_predictions/` (extension included is fine; will be stripped) |
| `x` | int | yes | pixel column of any point inside the cell |
| `y` | int | yes | pixel row of any point inside the cell |
| `label` | enum | yes | `sickle` \| `non_sickle` \| `ambiguous` (`ambiguous` rows are excluded from training) |
| `annotator` | str | optional | initials or name |
| `notes` | str | optional | free-form |

## Example

```csv
source_image,x,y,label,annotator,notes
roi_001.jpg,412,873,sickle,UG,
roi_001.jpg,1204,556,non_sickle,UG,
roi_002.jpg,733,89,ambiguous,UG,partially out of frame
```

## Behavior on resolution

* If `(x, y)` falls inside an instance footprint → that instance gets the label.
* If it falls in background or polymer → row goes to `failed.jsonl` with reason `coordinate_outside_cell`.
* If it falls inside a dropped instance (edge-touching, below `min_area`, above `max_area`) → row goes to `failed.jsonl` with reason `instance_dropped`.
* Multiple coordinates resolving to the same instance → first wins, others go to `failed.jsonl` with reason `duplicate_instance`.
