# Kaggle preprocessing target rebuild audit — 2026-11-13

## Counts and boundary

- Raw competition train: 1,920 source photos.
- The active processed manifest has 1,906 full-frame train rows plus seven
  crop rows; 1,886 samples are eligible and 27 full-frame rows are held out.
- The frozen native Kaggle pipeline was compared posthoc on 1,870 full-frame
  IDs shared with the active reference plus confirmed seed corrections: 848
  match and 1,022 differ. Fifty raw IDs have no comparable eligible target
  entry (47 lack a target entry; three target IDs are not eligible full frames).

## Why the current Kaggle notebook diverges

It uses a separate EfficientNetV2-S + HSV descriptor + 96-cluster pipeline,
then forces nearly uniform class counts. The active processed assignments use
the initial fixed hierarchical routing chain, subsequent train-only
refinements, and participant seed updates. The active manifest also has 56
full-frame assignments after the NPZ snapshot was last created: 51 from the
1039 train-only cluster, four participant-confirmed seed updates, and one
geometric source-group vote. The itemized delta is in
`processed_manifest_vs_npz_delta.csv`.

The seed-only and multi-cluster probes were independent train-only experiments;
neither had adequate held-out seed performance to replace the active chain.
This audit does not use active labels for fitting or tuning. No active label,
Kaggle notebook, or training target was changed here.

## Rebuild implication

Adding more arbitrary clusters to the current notebook will not reproduce the
active assignments. A faithful dataset-only rebuild needs to port the active
initial routing and accepted train-only refinement methods, regenerate their
features inside the notebook, then apply only the documented participant seeds,
crop policy, exclusions, and deterministic source-group rule. Classifier
training stays gated until that run has a verified parity report.


## Active assignment source inventory (13 Nov)

The notebook seed mapping was checked against `data/processed/manual_seeds.csv`: all 37 IDs/classes match. The active manifest has 1,879 eligible full-frame rows. Of these, 1,555 originate from `iterative_program_refinement_snapshot`; the other eligible rows span twelve additional provenance sources. The current Kaggle notebook still does not rebuild any of these assignments; its label cell intentionally fails closed.

`active_assignment_source_inventory.csv` and `active_assignment_id_provenance.csv` record the source counts and per-ID provenance without exporting class targets. This is a dependency inventory only, not a parity result. Exact parity still requires porting and rerunning the baseline and all subsequent train-only refinements from raw images, then performing a frozen per-ID audit. Crops and classifier training remain gated.

## Status replay notebook Kaggle (audit lanjutan 13 Nov)

Notebook `final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb` saat ini:

- Menginventarisasi 1.920 gambar train dan membuat 37 seed tim dari raw dataset.
- Menghitung DINOv2 global/patch dan DINOv3 Small descriptor dari raw train.
- Menjalankan eksperimen hybrid CNN+HOG+warna terpisah; OOF 37 seed sangat rendah (accuracy CNN 2,7%, HOG+warna 0%, gabungan 0%). Eksperimen tersebut tidak menghasilkan assignment aktif dan tidak boleh dipakai untuk menyimpulkan kategori train.
- Belum menghitung bank Task8 yang digunakan oleh refinement iteratif lima-keluarga.
- Belum membangun baseline Stage17. Stage17 memakai head terlatih Stage12/Stage16 dan coarse head Stage4; sumber historis tersedia di arsip lokal, tetapi notebook tidak melatih ulang rantai Stage4→12→16 dari raw.
- Belum menjalankan tahap konsensus awal dan refinement berurutan yang menghasilkan populasi aktif: 1.906 full-frame rows, 1.879 eligible, 27 ditahan, dengan 14 kelompok provenance algoritmis/manusia yang tercatat dalam manifest.

Karena itu, tidak ada prediksi full-frame hasil replay untuk dibandingkan per-ID dengan manifest aktif. Existing parity dari rebuild v1 hanya membuktikan reproduksi assignment awal dengan cache fitur/checkpoint arsip; ia bukan parity v3 dan bukan raw-only replay. Cell parity tetap `False`, crop dan classifier tetap dilewati.

Langkah implementasi yang masih diperlukan: (1) pindahkan kode pelatihan pembentuk head Stage4/12/16 ke notebook dan ekstraksi seluruh fitur pendukung dari raw train; (2) hasilkan baseline Stage17; (3) port rangkaian refinement beku sesuai urutan/provenance aktif, termasuk Task8 dan aturan peserta; (4) bekukan assignment, hitung join satu-ke-satu per ID dan laporan selisih; (5) buka gate hanya bila seluruh ID dan kategori cocok dengan manifest aktif. Tidak boleh mengimpor tabel/NPZ assignment sebagai input atau menyetel metode memakai hasil perbandingan.

## Stage17 baseline versus active full-frame manifest (posthoc diagnostic)

The historic Stage17 output has 1,920 train IDs. Its source hashes match the active manifest on all 1,906 shared full-frame IDs. Before any refinement, Stage17 agrees with 1,679/1,906 active assignments (1,666/1,879 eligible). Applying only the 37 participant-confirmed seeds raises this to 1,707/1,906 (1,694/1,879 eligible), leaving 199 full-frame label differences (185 eligible). The 37 seed labels independently agree with the active manifest; they were not read from the active manifest for this calculation.

The remaining 199 differences are concentrated in provenance families: 67 `vit_knn_label_propagation_snapshot`, 38 `iterative_program_refinement_snapshot`, 33 `augmented_source_prototype_snapshot`, 17 `bounded_anchor_stable_consensus_snapshot`, 16 `residual_graph_correction_snapshot`, 12 `two_view_lighting_ontology_snapshot`, six each in `human_seed_source_concept_snapshot` and `bounded_anchor_cluster_consensus_snapshot`, two each in `retired_anchor_holdout` and `robust_neural_assignment_snapshot`. This decomposition is for audit only; these target labels were not used for fitting or selection.

Per-ID baseline detail: `stage17_baseline_vs_active_manifest_per_id.csv`; seed-overlay detail: `stage17_baseline_plus_seeds_vs_active_manifest_per_id.csv`. These compare a historical baseline artifact with the active manifest and do **not** constitute the requested raw-only notebook replay. The notebook still needs to rebuild Stage17 and replay these methods from raw images before its parity gate can pass.


## Train-only semantic anchor replay progress (13 Nov)

After the raw semantic feature banks, the Kaggle notebook now derives a deterministic three-family weak-consensus anchor bank from ResNet recycling, EfficientNet recycling, and ConvNeXt ImageNet-22k ontology evidence. The rule requires at least two reliable source votes, with per-source conditional confidence >=0.8 and coverage thresholds 0.6/0.6/0.2. It records every source probability/coverage, vote, mask, class order, raw source hashes, and class counts under the Run All working directory. It reads no assignment NPZ, reference labels, or test pixels.

This is an upstream anchor candidate only. It is not the four-source Stage16 reproduction (the historical generic material prior is still missing), does not train Stage12/16 student heads, does not generate the active full-frame assignment, and has not been executed in Kaggle. Crops and classifier training remain gated; parity is still unverified.

## Raw-only replay implementation progress (13 Nov, after webinar revision)

The Kaggle notebook now has a train-only Stage4 replay cell following the raw DINOv2 feature extraction. It builds exact/pHash source groups from the 1,920 raw train images only, makes a grouped 80/20 fit/holdout split, and fits the SCAN-inspired anonymous 8-cluster consistency head with the recorded seed/entropy search. It exports the cluster IDs, probabilities, fitted head, and run report under the notebook working directory. No semantic category labels are assigned in this step, and no test images enter its feature extraction, grouping, split, or fit.

The raw-only Stage4 cell now serializes both `fit_ids` and `heldout_ids`, their train-only source-group IDs, and the unique train IDs used to refit the coarse head. This makes the same train-only partition available to subsequent head-fitting cells without consulting an assignment cache. The notebook edit passed JSON/Python syntax checks only; the checkpoint has not yet been generated by a Kaggle run.

This is a concrete prerequisite for the Stage17 routing baseline, not a parity result. The original split used source groups constructed over the full image inventory, whereas this replay deliberately uses train-only groups to avoid letting test images affect training. That may change the coarse split and downstream labels. Stage12/16 semantic heads, the full Stage17 route, and accepted v3 train-only refinements are still not rebuilt from raw in the notebook. `LABEL_PARITY_VERIFIED` remains `False`, so processed class folders, crops, and classifier training remain gated. The new cell has only passed static Python syntax/JSON checks; it has not been executed on Kaggle/GPU, and no per-ID parity file exists for this replay.


## Hybrid CNN+HOG+color labeling candidate (13 Nov, local CPU)

The Kaggle notebook includes the webinar feature-fusion experiment as a separate raw-train-only candidate stage: frozen EfficientNet-B0 embedding + HOG + RGB histogram, grouped nested OOF over 37 participant-reviewed train seeds, and RBF-SVM. The candidate stage does not write processed labels and uses zero test pixels. Local execution completed from the raw train set with a hash-verified feature cache. Hybrid grouped OOF was 3/37 (8.1%; macro-F1 0.119), below CNN-only at 4/37 (10.8%; macro-F1 0.104); the full-train candidate counts were strongly skewed. Therefore it is retained as an experiment but rejected as an active train-label rule. It does not resolve assignment parity. Details and artifacts are in `HYBRID_LABELING_RESULT.md` and the sibling `hybrid_local_run/` directory.


## Raw-only Stage8 semantic-anchor replay port (13 Nov)

A new notebook stage now ports the historical Stage8 dependency chain from raw-train-derived evidence: it aligns Stage4 anonymous clusters by ID, fits the cluster-to-category Hungarian mapping on Stage4 fit IDs only, constructs confidence/source-consensus anchors from the notebook-generated generic and recycling-CNN priors, then fits linear/MLP heads over ResNet-waste, DINOv2, and fused features with fit-only neighbor/view regularization. It serializes its own checkpoint, full-train probabilities, anchor counts, source fingerprint, and split provenance under the notebook working directory. It reads no assignment cache, archive labels, or test pixels.

At the time of this entry, only the Stage8 cell had been ported; it was static and had not run on Kaggle/GPU. Assignment parity was unverified and the processed-label/classifier gate remained closed. The later notebook revision is recorded below.

## Raw-only replay port update (13 Nov, later notebook revision)

The notebook now also contains static train-only ports for Stage9 consensus-anchor expansion, Stage12 ontology-expert refinement, Stage16 independent-source consensus, and Stage17 fixed group routing after Stage12/16. They consume feature banks and Stage4 fit/held-out groups generated by preceding notebook cells; they do not read assignment NPZs, active-label tables, archive annotations, or test pixels. They write intermediate probabilities and provenance under the notebook working directory.

These four cells have only been reviewed/compiled for syntax in the notebook source; they have not been executed end-to-end on Kaggle. Consequently there is no raw-only full-frame assignment or per-ID parity report yet. The replay gate remains false, and the seven crop rows, processed class folders, and classifier-training cells remain skipped. This is implementation progress, not proof of reconstruction parity.

## Stage18 raw-only iterative assignment candidate (13 Nov, later notebook revision)

The Kaggle notebook now adds a five-family cross-fit denoising stage after its raw-only Stage17 candidate. It uses the pinned recipes (DINOv2 global linear SVM, DINOv2 local kNN-7, DINOv3 global kNN-21, DINOv3 foreground RBF-SVM, and DINOv3 plus the eight-class CNN embedding with linear SVM), source-grouped five-fold predictions, a four-of-five change rule, fixed high-consensus anchors, and a 1% stopping condition. The 37 recorded team seed labels are applied afterward as an explicit overlay. The stage writes a 1,920-row candidate table and per-ID provenance; it reads no active assignment, assignment NPZ, archive label file, or test image.

The eight-class teacher checkpoint is downloaded at a pinned revision and parsed as inert tensor data. The notebook does not unpickle/execute checkpoint callables. The extraction/model-load helper was exercised locally against the cached 103 MB checkpoint: all 330 tensors loaded strictly into the ResNet50 teacher and the model returned eight logits. The notebook passes nbformat validation and Python syntax parsing (50 cells, 25 code cells, no syntax errors); these checks do not replace a full runtime.

Stage18, its raw feature extraction, and the complete Stage4-to-Stage18 assignment replay have not yet run end-to-end in Kaggle. No candidate per-ID parity output has been produced. `LABEL_PARITY_VERIFIED` remains false; processed folders, crops, and classifier training stay gated. The candidate is therefore an implementation step, not proof of parity or semantic correctness.


## Stage17 + seed audit port (13 Nov 2026)

`final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb` now writes the raw-train Stage17 assignment and a separate `stage17_plus_37_seed_audit_assignments.csv` / report under the run output. The 37 participant-provided labels are overlaid only in that audit copy; they do not change Stage16/17 probabilities, fitting, or model selection. The Kaggle notebook does not load the active assignment manifest, so it cannot report agreement counts such as 1,707/1,906; those remain results of the external post-hoc audit.

Static validation: notebook JSON/nbformat and code-cell AST checks pass. Full Stage17 rerun was not completed locally: CPU replay exceeded practical runtime during semantic feature extraction. The notebook's assignment parity gate remains closed until a full run and independent per-ID audit are completed.

The later Stage18 cross-fit report also labels its 37-seed column `audit_seed_overlay_class`; the automatic assignment remains `automatic_refined_class`. The audit overlay is not an input to cross-fit or any model-selection step.


## Full Stage17 CPU replay result (14 Nov 2026)

The assignment-only replay completed locally on CPU using the raw dataset and notebook-generated caches. It produced **1,639/1,906** Stage17 matches and **1,664/1,906** after the separate 37-seed audit overlay, versus the historical audit's 1,679 and 1,707. The current replay differs from the historical Stage17 assignment on 132 shared IDs despite all 1,906 source hashes matching. Full per-ID outputs and reports are preserved in `stage17_cpu_replay_20261114/`. The final classifier/submission cells were not run; parity gate remains closed.


## Further train-only experiment (14 Nov 2026)

Stage18 five-family grouped cross-fit changed 108 automatic labels from Stage17. In the post-hoc active-manifest audit, the automatic assignment matched 1,684/1,906 and the separate 37-seed audit overlay matched 1,712/1,906 (194 differences); on 1,879 eligible rows, the overlay matched 1,698 (181 differences). This was the fixed Stage18 procedure, not a model/parameter search against the manifest. Artifacts are under `stage18_cpu_refinement_20261114/`.

A separate screen of 30 seed-supervised expert recipes used leave-source-group-out OOF on the 37 participant labels, with no active-manifest input. Best was DINOv2 cosine 1-NN at 13/37 (35.1%) and macro-F1 0.353. That signal is too weak to replace the Stage18 assignment. Results are under `seed_oof_experts_20261114/`.
