# `batch_classify.ipynb` — output schema

Reference for every column in every output file produced by
[`notebooks/batch_classify.ipynb`](batch_classify.ipynb), with suggested
downstream analyses.

All files land directly under `INPUT_FOLDER/` (the condition-organised root),
except figures which go to `INPUT_FOLDER/figures/`.

---

## `per_fov.parquet` — one row per image

| column | type | description | useful for |
|---|---|---|---|
| `condition` | str | condition folder name | grouping for every plot |
| `image_name` | str | raw filename | drill-down to a specific FOV |
| `stem` | str | filename without extension | joining to `polymer_blobs`, `per_cell`, QA images |
| `n_sickle` | int | classifier-positive cell count | numerator of `frac_sickle` |
| `n_non_sickle` | int | classifier-negative cell count | denominator |
| `n_cells` | int | total classified cells | per-FOV cell density / yield QC |
| `mean_p_sickle` | float | mean softmax p(sickle) across cells in this FOV | soft fraction — less noisy than `frac_sickle` at low prevalence; smoother violins |
| `frac_sickle` | float | `n_sickle / n_cells` | the existing main metric |
| `polymer_length_um` | float | sum of kept-blob major axes (µm) | total protrusion burden per FOV |
| `n_polymer_blobs_kept` | int | blobs surviving the filter | sanity check — should track protrusion length |
| `n_polymer_blobs_dropped_too_short` | int | <`MIN_LENGTH_PX` | usually noise; high counts → segmentation churn |
| `n_polymer_blobs_dropped_too_long` | int | >`MAX_LENGTH_PX` | microscope artifacts; spike per FOV = bad image |
| `n_polymer_blobs_dropped_too_far` | int | >`MAX_DIST_FROM_CELL_PX` from any cell | dust, debris, background false positives |
| `polymer_area_fraction` | float | `kept_polymer_px / (kept_polymer_px + cell_tissue_px)` | bulk polymerization measure; complements length (length captures filaments, area captures dense blobs) |
| `polymer_skeleton_length_um` | float | total pixels of `skeletonize(kept_polymer_mask)` × scale | curved/branched fibers correctly summed; major-axis underestimates them |
| `polymer_endpoints` | int | skeleton pixels with 1 neighbor | fiber-tip count; rough fiber count |
| `polymer_branch_points` | int | skeleton pixels with ≥3 neighbors | network complexity; branching = more advanced polymerization |
| `polymer_um_per_100_cells` | float | `polymer_length_um × 100 / n_cells` | **legacy — not reported in the paper.** Uses all cells (sickle + non-sickle) in the denominator. The paper's Figure 2e uses per-sickle-cell only (`polymer_length_um / n_sickle`, pooled per condition — see `notebooks/analysis_protrusion_per_condition.ipynb` cell `Idea O`). |

> **Paper metric (Figure 2e):** `sum(polymer_length_um) / sum(n_sickle)` pooled across FOVs per condition, with a 1000-resample bootstrap CI. Computed by `notebooks/analysis_protrusion_per_condition.ipynb` (Idea O). Neither `polymer_um_per_100_cells` nor a per-FOV distribution of `polymer_length_um / n_sickle` are what the paper reports.

**Analysis ideas** (methodology exploration — not reported in the paper):
`polymer_endpoints / 2` ≈ fiber count, then
`polymer_length_um / fiber_count` ≈ mean fiber length per FOV. Branch-points ÷
endpoints ratio captures network topology and may separate conditions that
look similar on length alone.

---

## `per_condition.parquet` — one row per condition

Sums + medians from `per_fov` and `per_cell`. Drop-in for tables in a
manuscript.

| column | type | description |
|---|---|---|
| `condition` | str | condition name |
| `n_images` | int | FOVs in this condition |
| `n_sickle`, `n_non_sickle`, `n_cells` | int | grand totals |
| `polymer_length_um_sum`, `polymer_skeleton_length_um_sum`, `polymer_endpoints_sum`, `polymer_branch_points_sum` | float/int | grand polymer totals |
| `n_polymer_blobs_kept_sum`, `n_polymer_blobs_dropped_too_*` | int | grand blob counts |
| `median_frac_sickle`, `median_mean_p_sickle`, `median_polymer_um_per_100_cells`, `median_polymer_area_fraction` | float | median across FOVs |
| `median_eccentricity`, `median_axis_ratio`, `median_solidity`, `median_compactness`, `median_n_convexity_defects`, `median_max_defect_depth_um` | float | median across cells |

**Analysis ideas** (methodology exploration — not reported in the paper):
Sort by `median_polymer_um_per_100_cells` to rank conditions by
polymerization burden. The paper's Figure 2e uses a different (pool-level,
per-sickle-cell) ranking; see the top of this file.

---

## `per_cell.parquet` — one row per classified cell

| column | type | description | useful for |
|---|---|---|---|
| `condition`, `image_name`, `stem` | str | grouping keys | grouping, joining |
| `row_idx` | int | global row index | joins to `per_cell_morphology.pt` |
| `instance_id` | int | watershed label in this FOV's instance image | joining to `polymer_blobs.assigned_instance_id` |
| `predicted_label` | str | `"sickle"` or `"non_sickle"` | filter for sickle-only analyses |
| `p_sickle` | float | classifier softmax for sickle class | calibration plots, severity-by-confidence stratification, threshold sweeps without rerunning the model |
| `assigned_polymer_length_um` | float | sum of major µm of blobs assigned to this cell | per-cell polymer burden; doesn't saturate like a ratio |
| `area_px` | int | cell area | size distributions per condition — sickle cells can be smaller |
| `eccentricity` | float | 0=circle, →1=line; from inertia tensor | classical sickle elongation marker |
| `axis_ratio` | float | major / minor axis | complementary elongation metric, less sensitive to noise than eccentricity at low values |
| `solidity` | float | area / convex_hull_area | spike/protrusion proxy; low solidity = irregular outline |
| `compactness` | float | `perimeter² / (4π · area)` | 1=circle, ↑=jagged perimeter; spike/roughness proxy |
| `n_convexity_defects` | int | count of inward concavities with depth ≥ `DEFECT_DEPTH_MIN_PX` | classifier-free sickle proxy; sickle typically 2+, smooth disc 0 |
| `max_defect_depth_um` | float | deepest concavity in µm | distinguishes deep crescent notches (sickle) from shallow ripples (echinocyte) |

**Analysis ideas:**
- **Calibration plot:** bin cells by `p_sickle`, plot mean `n_convexity_defects` per bin — agreement validates both
- **2-D shape map:** `eccentricity` vs `solidity` scatter, colored by condition — natural sickle / discocyte / echinocyte separation
- **Identify edge cases:** cells with high `assigned_polymer_length_um` but low `p_sickle` → classifier misses for manual review
- **Stratify severity:** among predicted-sickle cells, split by `assigned_polymer_length_um` quartile and compare conditions
- **Echinocyte vs sickle:** pair `(n_convexity_defects, max_defect_depth_um)` — many shallow defects = echinocyte; few deep defects = sickle

---

## `per_cell_morphology.pt` — torch.save dict, row-aligned with `per_cell.parquet`

| key | type | description |
|---|---|---|
| `features` | `Tensor[N, 30]` float32 | the raw morphology vector seen by the classifier's morphology tower |
| `feature_names` | `list[str]` of length 30 | column names: `area`, `perimeter`, `compactness`, `eccentricity`, `solidity`, 8 Fourier descriptors, 16 Zernike moments, `polymer_ratio` |
| `row_idx` | `Tensor[N]` int64 | join key against `per_cell.parquet.row_idx` |

**Analysis ideas:** Fourier descriptors capture multi-scale shape regularity
that summary metrics miss — useful for PCA / UMAP of shape across conditions.
Zernike moments are rotation-invariant; clustering cells in Zernike space
sometimes yields cleaner morphology subtypes than thresholding individual
scalars. Skip loading this file unless you specifically need the 30-D
vector — `per_cell.parquet` has the named summaries.

---

## `polymer_blobs.parquet` — one row per polymer connected component

| column | type | description | useful for |
|---|---|---|---|
| `condition`, `image_name`, `stem` | str | grouping keys | grouping, QA lookup |
| `blob_id` | int | label in this FOV's polymer connected-component image | joining back to `instance_image` for visualization |
| `major_um` | float | major axis length in µm | the violin plot input (kept blobs only) |
| `minor_um` | float | minor axis length in µm | fiber thickness; thin fibers vs bulk polymer |
| `area_px` | int | blob pixel count | filtering / sizing |
| `eccentricity` | float | 0=disc, →1=line | distinguishes thin fibers (high) from compact bulk (low) |
| `solidity` | float | blob's area / convex hull area | branched / forked fibers have lower solidity |
| `dist_to_cell_px` | float | min distance from any blob pixel to cell tissue | filter knob diagnostic; biological "polymer leakage" if you trust it |
| `assigned_instance_id` | int / null | which cell this blob is attributed to (None = orphan) | per-cell polymer aggregation; orphan rate = QC metric |
| `kept` | bool | passed all three filters | violin/severity/area calcs use `kept == True` |
| `drop_reason` | str | `""`, `"too_short"`, `"too_long"`, `"too_far"` | QA figure colors, threshold tuning |

**Analysis ideas:**
- **Fiber-length violin per condition** (already generated) — log-y because lengths span 2+ decades
- **Aspect ratio:** `minor_um / major_um` per condition — thin filaments vs chunky polymer
- **Orphan rate:** fraction with `assigned_instance_id is None` per condition — high orphan rate flags imaging or segmentation problems
- **Drop-reason histograms:** if any condition has unusual `too_long` rates, that's a microscopy QC signal worth checking before drawing biological conclusions

---

## `pairwise_stats.parquet` — pairwise condition comparisons

| column | type | description |
|---|---|---|
| `metric` | str | which metric was tested (11 total: see below) |
| `condition_1_index`, `condition_2_index` | int | 1-based indices into the alphabetical condition list |
| `condition_1_name`, `condition_2_name` | str | names |
| `p_value` | float | raw two-sided Mann-Whitney (or other if changed) |
| `p_value_fdr` | float | BH-corrected within this metric across all pairs |
| `test` | str | which test was used |
| `fdr_method` | str | `"fdr_bh"` or `"None"` |

**Metrics tested** (5 FOV-level + 6 cell-level-medianed-to-FOV):

- Per-FOV: `frac_sickle`, `mean_p_sickle`, `polymer_um_per_100_cells`,
  `polymer_area_fraction`, `polymer_skeleton_length_um`
- Per-cell medianed per FOV: `median_eccentricity`, `median_axis_ratio`,
  `median_solidity`, `median_compactness`, `median_n_convexity_defects`,
  `median_max_defect_depth_um`

**Analysis ideas:** for the paper, report `p_value_fdr` (not raw) and cite
n_images per condition from `per_condition.parquet`. Filter to
`metric == "frac_sickle"` for the main result table.

---

## `manifest.json` — provenance

Plain dict with `timestamp_utc`, `input_folder`, `checkpoints` (paths +
SHA-256 for both U-Net and classifier — proves which weights produced these
numbers), `config` (all thresholds), `conditions` (per-condition n_images +
n_cells), `outputs` (paths), `failed_count`. Drop into supplementary methods
as-is. If a reviewer asks "what was `POLYMER_MAX_DIST_FROM_CELL_PX` when you
made Figure 4?", this is the answer.

---

## `failed.jsonl` — error log

One JSON object per line: `{condition, image, error}`. Empty file = clean
run. Worth grepping after a full run to see if any conditions are
systematically failing (e.g. one condition's filenames have spaces and a
downstream function chokes).

---

## Quick join reference

```
per_cell.parquet[row_idx]        <-->  per_cell_morphology.pt[row_idx]
per_cell.parquet[instance_id]    <-->  polymer_blobs.parquet[assigned_instance_id]
                                       (within same condition+stem)
per_cell.parquet[stem]           <-->  per_fov.parquet[stem]
per_fov.parquet[condition]       <-->  per_condition.parquet[condition]
polymer_blobs.parquet[stem]      <-->  figures/polymer_qa/<condition>_<stem>.png
```
