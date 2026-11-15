# Lunar Terrain Analogs

**Which terrestrial landscapes resemble the lunar south pole, which lunar terrain has no
terrestrial counterpart, and where does Saudi Arabia fit?**

NASA Space Apps Challenge 2026 — *Identify Earth Locations that Analog the Permanent Moon
Base Locations and Mars*. Solo entry, Universal Event.

**Interactive results:** https://reemfaisal1.github.io/lunar-terrain-analogs/

---

## Summary

A convolutional autoencoder was trained on 121,914 elevation patches from the lunar south pole
and 27 sites on Earth, with no labels and no information about which body each patch came from.
Matching in its learned representation shows that:

- **Earth is about half as good a source of lunar south-pole terrain as the Moon itself**
  (0.54×), and its usefulness falls steadily with relief — from 0.70× for gentle terrain to
  **0.02× for steep ridges and crater rims**, which the Moon matches in itself fifty times more
  often than Earth does.
- **The representation encodes landform shape that slope does not.** It predicts all-direction
  terrain enclosure under a low sun far better than slope and relief statistics, though it adds
  nothing for slope-driven traversal cost.
- **It ranks established analog sites above controls** (site-ranking AUC 0.786 against 0.643 for
  hand-built features), and the ranking survives adjustment for site size.
- **The Artemis III candidate regions richest in rim terrain are the hardest to rehearse on
  Earth**, from 8.7% matched at de Gerlache to 40.1% at Haworth.
- **In Saudi Arabia, the Tuwayq escarpment carries lunar terrain; the sand seas do not.**

## The question

Analog sites — Haughton Crater, the Atacama, the Antarctic Dry Valleys — are chosen for mission
rehearsal largely by expert judgement. This project asks a narrower, testable question:

> Can a model trained **only on elevation**, with no labels and no knowledge of which body it is
> looking at, find terrestrial terrain that shares measurable structure with lunar south-polar
> terrain — and can we verify that what it learned reflects real physical properties rather
> than data artefacts?

Scope is topography and terrain-driven illumination only. This is not a geological, thermal or
volatile analog assessment, and not a site-selection recommendation.

---

## Results

### 1. How good a source of lunar terrain is Earth?

A lunar patch counts as **matched** when some patch is at least as close to it, in the learned
64-dimensional space, as a held-out lunar patch typically is to its nearest lunar neighbour in
*other* regions (threshold T ≈ 1.96). By construction, the Moon therefore matches its own
held-out terrain half the time — so Earth's coverage must be read against that baseline, not
against 100%.

Earth's reference pool (66,218 patches) is larger than the lunar one (44,976), and larger pools
give closer nearest neighbours, so the comparison below cuts the Earth pool to the lunar pool's
size (`src/baseline_check.py`, on 10,720 held-out lunar patches).

| Regime | Share of pole | Relief | Moon matches itself | Earth matches it | Earth relative to the Moon |
|---|---|---|---|---|---|
| R1 gentle terrain | 46% | 156 m | 64.2% | 45.0% | **0.70×** |
| R0 moderate relief | 28% | 219 m | 40.3% | 16.5% | **0.41×** |
| R3 basins and depressions | 14% | 280 m | 47.4% | 7.6% | **0.16×** |
| R2 steep ridges and rims | 12% | 401 m | 18.1% | 0.3% | **0.02×** |
| **All held-out terrain** | | | **50.0%** | **26.8%** | **0.54×** |

Earth's usefulness falls monotonically with relief. R2 is not simply unusual terrain that
nothing matches: other lunar regions still supply a match for it 18.1% of the time, against 0.3%
for Earth. The Earth sites rugged enough to qualify — the Alps, the Himalaya — are rugged in
the wrong way, carved by ice rather than built by impact.

This describes **the 27 sampled sites**, not Earth. Intermediate-relief terrain may exist and
simply not be sampled here.

The four regimes come from k-means on the lunar embeddings (k = 4, chosen by silhouette subject
to split-half stability ARI ≥ 0.6). Silhouette is only 0.26, so they are **four stable positions
along a continuum**, not discrete terrain types. Their descriptor profiles — which the model
never saw — rise steadily in slope and relief, and R3 carries the highest terrain enclosure
(0.06) without the highest slope.

### 2. The representation captures landform shape, not just steepness

Physical descriptors computed independently of the model were predicted from the embedding on
spatially held-out regions (gradient-boosted regression, 2,290 training / 600 test patches):

| Target | Features | R² (both bodies) | R² (Moon only) |
|---|---|---|---|
| Terrain enclosure (`shadow_always`) | slope/relief only | 0.259 | 0.065 |
| | embedding only | 0.406 | 0.315 |
| | **both** | **0.425** | **0.330** |
| Traversal cost (`route_ratio`) | **slope/relief only** | **0.723** | 0.830 |
| | embedding only | 0.609 | 0.763 |
| | both | 0.637 | **0.841** |

Adding the embedding to slope and relief raises enclosure R² by **+0.198 (95% CI +0.064 to
+0.338, spatial-block bootstrap over 327 blocks, P(gain > 0) = 0.998)**, measured across both
bodies on 626 test patches (`src/checks.py`). On the Moon alone the effect is larger: 0.07 from
slope and relief, 0.33 with the embedding.

Enclosure in all 16 azimuths requires concavity — a uniform slope always has a sunlit downhill
direction — so this is evidence that the representation encodes landform shape. It was never
trained to predict enclosure, only to rebuild elevation patches.

For traversal cost, which is computed directly from slope, the embedding adds nothing over
slope and relief. That is reported rather than hidden.

### 3. Known analog sites rank above controls

For each Earth site, the score counts **distinct lunar patches** with a match from that site
among their five nearest Earth neighbours, per 1,000 of the site's own patches. It has to run
from the Moon outward: scored the other way, a uniform plain trivially finds a lunar neighbour
for every one of its own patches. An earlier Earth-centric version scored the flat control
site and Rub' al Khali at 100%, and the control test caught it.

| Method | Site-ranking AUC (7 known analogs vs 14 control/background) | Permutation *p* |
|---|---|---|
| **Learned embedding** | **0.786** | **0.022** |
| Hand-built slope/relief features (identically calibrated threshold) | 0.643 | 0.299 |

- **Controls** rank 19th (Great Smoky Mountains) and 23rd (Great Plains) of 27.
- **Threshold sensitivity**: ranking stable at ±20% on T (Spearman 0.94–0.98, AUC 0.72–0.79).
- **Site size**: known analogs average fewer patches (1,843 vs 2,413) and size correlates weakly
  with the score (Spearman −0.23). After regressing out log site size the enrichment persists —
  AUC 0.72, *p* = 0.031 — and the largest known analog, Atacama, still ranks third.

This is a **retrospective site ranking**, not held-out predictive accuracy: labels never enter
training, but the model has seen terrain from the sites whose labels score the ranking.

### 4. Artemis III candidate landing regions

Six regions from NASA's candidate list, each approximated by the USGS/IAU gazetteer disk of the
feature it is named after (the projection was checked against true pole distance to within
0.1 km). The more R2 terrain a region contains, the less of it Earth supplies:

| Region (approximated by) | Latitude | R2 share | Matched | Leading Earth sources (enrichment) |
|---|---|---|---|---|
| de Gerlache Rim (de Gerlache crater) | 88.5°S | 28.9% | **8.7%** | Atacama 4.7×, Death Valley 4.2×, Etna 3.7× |
| Nobile Rim (Nobile crater) | 85.3°S | 20.4% | 15.2% | Atacama 3.6×, Hawaii 3.3×, Hekla 3.1× |
| Malapert Massif (Mons Malapert) | 86.0°S | 14.3% | 26.0% | Atacama 3.8×, Hekla 3.1×, Hawaii 2.9× |
| Peak near Shackleton (Shackleton crater) | 89.9°S | 24.6% | 26.1% | Etna 5.5×, Haughton 3.2×, Atacama 3.1× |
| Faustini Rim A (Faustini crater) | 87.2°S | 15.0% | 33.0% | Hawaii 6.7×, Atacama 3.8×, Death Valley 2.7× |
| Haworth (Haworth crater) | 86.9°S | 8.9% | **40.1%** | Atacama 3.5×, Hawaii 3.4×, Hekla 2.7× |

For comparison, the whole south pole is matched at 27.3% on the same measure. Enrichment divides
a site's share of matches by its share of the Earth sample, separating similarity from sample
size. Atacama leads at 3–5× its share: the method independently recovers the most heavily used
terrestrial analog without being told it is one. With six regions this is a visible pattern, not
a statistically established correlation.

### 5. Saudi Arabia: the escarpments, not the sand seas

| Site | Rank of 27 | Lunar coverage per 1,000 |
|---|---|---|
| Tuwayq escarpment | 8 | 1,069 |
| Asir mountains | 13 | 684 |
| Harrat Rahat | 18 | 451 |
| *Great Smoky Mountains — control* | *19* | *442* |
| Harrat Khaybar | 20 | 417 |
| *Great Plains — control* | *23* | *184* |
| Rub' al Khali | 25 | 84 |
| Wahbah crater | 26 | 77 |

Tuwayq sits between Death Valley and Askja, and Asir above both controls; both sand-sea sites
fall below the flat control. The method's useful output is discrimination, not endorsement.

---

## Data

| Source | Use | Coverage |
|---|---|---|
| **LOLA LDEM 80S**, 80 m/px, south polar stereographic | lunar elevation | 80–90°S |
| **ASTER GDEM v3** | Earth elevation, polar sites | 83°N–83°S |
| **NASADEM HGT** | Earth elevation, all other sites | 60°N–56°S |

Earth data is retrieved with `earthaccess` (NASA Earthdata Login required). No elevation data is
committed; `src/download_earth.py` and `src/build_grids.py` regenerate it.

**27 Earth sites**, each labelled with a role used only for validation, never for training:

- **Known analogs (7):** Haughton, Antarctic Dry Valleys, Askja, Atacama, Hawaii, Death Valley,
  Craters of the Moon
- **Saudi Arabia (6):** Harrat Khaybar, Harrat Rahat, Wahbah, Asir, Tuwayq, Rub' al Khali
- **Controls (2):** Great Smoky Mountains, Great Plains
- **Background (12):** Alps, Himalaya, Tibesti, Ahaggar, Sinai, Grand Canyon, Andes, Namib, Gobi,
  Etna, Svalbard, Hekla

## Method

**Common grid.** Both bodies are resampled to 80 m/px, LOLA's native resolution. Earth tiles are
merged, reprojected to a local Lambert azimuthal equal-area projection centred on each site, and
area-averaged.

**Resolution matching.** Earth DEMs carry pixel-scale speckle that LOLA does not. Uncorrected,
the autoencoder learned the noise difference instead of the terrain: Earth validation loss was
0.41 against the Moon's 0.06. Gaussian smoothing of Earth patches at σ = 1.5 px matches the
measured high-frequency energy ratio (`src/noise_check.py`: Moon 0.166, smoothed Earth 0.157)
and brought Earth validation loss to 0.134.

**Patches.** 64 × 64 px (5.12 km), stride 32 → 121,914 patches (55,696 lunar, 66,218
terrestrial). Coastal sites are masked for water. Each patch has its **best-fit plane removed**,
so the model sees landform shape rather than elevation or regional tilt; input is compressed as
asinh(h/50) for training.

**Physical descriptors — the answer key** (`src/descriptors.py`, never shown to the model):

- Slope percentiles, roughness (RMS deviation from a 5×5 local mean), relief (p95 − p5). Because
  patches are plane-removed first, slope and relief are *residual* values about the regional
  trend; roughness is unaffected.
- **Terrain enclosure**: rays in 16 azimuths against a 1.5° sun, out to 9.6 km, with a curvature
  drop of d²/2R for each body's radius, computed on the **original DEM context window**.
  `shadow_always` is the fraction of pixels occluded in all 16 directions.
- **Traversal cost**: least-cost paths on a cost surface of 1 + (slope/10°)², impassable above
  20°, over 12 random pairs per patch.

Enclosure and traversal are computed for a random sample of 1,500 patches per body.

**Autoencoder** (`src/train_ae.py`). Four stride-2 convolutional blocks to a 64-dimensional
embedding, mirrored decoder. Loss = MSE + 10 × gradient-matching term, so terrain transitions are
preserved. Trained on balanced 64 + 64 Moon/Earth batches with rotation and reflection
augmentation, 4,000 steps (Adam, lr 1e-3 → 3e-4). Embeddings are averaged over the 8 dihedral
transforms, making them orientation-invariant. Final validation loss: Moon 0.063, Earth 0.134,
with train and validation curves tracking throughout.

There are no labels saying which Earth terrain is lunar-like, so supervised learning is not
available. Reconstructing a patch from 64 numbers forces the model to encode its shape, which
makes nearest-neighbour search in that space a shape comparison. The autoencoder is the means;
the validation is the contribution.

**Spatial validation.** Patches are grouped into 256 px (≈20 km) blocks, and 20% of blocks are
held out. The same split is used by every downstream analysis.

## Validation summary

| Question | Test | Result |
|---|---|---|
| Is coverage judged fairly? | Moon matched against itself, equal-sized pools | Moon 50.0%, Earth 26.8% → 0.54× |
| Did the model just learn which body it is? | Moon-vs-Earth classifier | 0.952 from the embedding, but **0.981** from five slope statistics — the gap is real terrain difference |
| Does it see beyond slope? | enclosure prediction, held-out regions | yes, +0.198 [0.064, 0.338]; not for traversal cost |
| Does it beat simple features? | site ranking | AUC 0.786 vs 0.643 |
| Is it just site size? | size-adjusted ranking | AUC 0.72, p = 0.031 |
| Do controls score low? | control ranks | 19th and 23rd of 27 |
| Does the threshold decide it? | ±20% on T | Spearman 0.94–0.98 |

## Limitations

1. **Topography only** — no geology, composition, regolith mechanics, thermal environment or
   volatiles.
2. **Terrain enclosure is not a permanently shadowed region** and says nothing about water ice.
3. **Traversal cost is a slope proxy**, not a rover mobility model.
4. **27 hand-picked Earth sites.** Coverage results are bounded by this sample.
5. **Seven known analogs** carry the enrichment test; several background sites (Hekla, Etna,
   Tibesti) are arguably analogs themselves, which makes the contrast conservative.
6. **Retrospective ranking**, not out-of-sample prediction.
7. **Model-relative threshold.** Coverage is only meaningful beside the lunar self-match baseline.
8. **Artemis regions are approximated** by each named feature's full disk, which is larger than
   NASA's ~15 km candidate region — for Peak near Shackleton, Shackleton crater itself.
9. **Block-boundary leakage.** Blocks are assigned by a patch's corner, so patches straddling a
   boundary can share area with the neighbouring block (roughly 7% of patches). This makes the
   threshold T slightly stricter, so coverage is if anything conservative.
10. **Baseline features see plane-removed patches**, like the model. Whether absolute regional
    tilt would narrow the enclosure gap is untested.
11. **Regime spatial organisation** was not resolvable in a 12 km binned density map
    (`src/fig_polar.py`); crater-rim-scale structure would need finer binning.
12. **A single training run.** Sensitivity to the random seed is untested.
13. **Mars is out of scope.** The pipeline runs on MOLA elevation unchanged, but was not run.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download_earth.py      # needs NASA Earthdata Login (~/.netrc)
# place the LOLA LDEM_80S_80MPP_ADJ GeoTIFF in data/raw/moon/
python src/build_grids.py         # reproject everything to the 80 m grid
python src/patches.py             # cut 121,914 patches
python src/descriptors.py         # physical descriptors (enclosure + traversal are slow)
python src/train_ae.py            # autoencoder; MPS or GPU recommended
python src/regimes.py             # regimes, threshold, matching, figures
python src/validation.py          # descriptor prediction, site ranking, baselines
python src/validation2.py         # domain-gap check, threshold sensitivity
python src/checks.py              # spatial-block bootstrap, site-size confound
python src/baseline_check.py      # lunar self-match baseline per regime
python src/artemis.py             # Artemis candidate regions
python src/export_web.py          # matched pairs for the web page
```

`results/ae.pt` holds the trained weights.

## Repository layout

```
src/        pipeline, in the order above
results/    CSV outputs, trained weights and figures
docs/       the interactive results page (GitHub Pages)
data/       not committed; regenerated by the download and build scripts
```

## References

- Barker, M. K., et al. (2023). Improved LOLA elevation maps for south pole landing sites.
  *The Planetary Science Journal*, 4, 183. doi:10.3847/PSJ/acf3e1
- NASA/METI/AIST/Japan Spacesystems and U.S./Japan ASTER Science Team. *ASTER Global Digital
  Elevation Model V003.* NASA LP DAAC. doi:10.5067/ASTER/ASTGTM.003
- NASA JPL. *NASADEM Merged DEM Global 1 arc second V001.* NASA LP DAAC.
  doi:10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001
- USGS/IAU. *Gazetteer of Planetary Nomenclature* — feature coordinates for the Artemis regions.
- NASA (2022, updated 2024). Candidate landing regions for Artemis III.

## Use of artificial intelligence

Machine learning is the project's core method: a convolutional autoencoder (PyTorch), k-means
clustering and gradient-boosted models (scikit-learn), all trained from scratch on public NASA
elevation data within this repository. No pretrained model, language model or external API is
part of the scientific pipeline.

Claude (Anthropic) was used as a coding and review assistant — for code scaffolding, debugging,
and critique of the validation design. All scientific decisions and claims are the author's, and
every result reported here is produced by the scripts in `src/`.

## Author

`Reem Alsahli` — NASA Space Apps Challenge 2026, Saudi Arabia (Riyadh Event).
