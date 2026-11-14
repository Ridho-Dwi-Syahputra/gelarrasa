# Rebuild audit: original preprocessing bundle v1

This run located and re-executed the original EDA/clustering stage and the
frozen hierarchical assignment stage. The exact v1 `preprocessing_inputs.npz`
was assembled again and compared array by array with the archived bundle.

## Results

- All 2,400 current raw image files match the source hashes recorded by the
  original run.
- Stage 1 reproduced the image profile, source groups, and ResNet50 feature
  cache byte for byte. All 2,400 anonymous cluster IDs match. The serialized
  cluster CSV differs only in floating-point text formatting in diagnostic
  columns; its cluster IDs are identical.
- The original fixed hierarchical assignment was rerun. Its CSV SHA-256 is
  `73996d8db5666bb0d03300f0204f54c1fddf39e8af77398dfc3b116664de6434`, exactly
  matching the frozen original assignment.
- The rebuilt v1 bundle matches the archived NPZ byte for byte:
  `5fb0e848c970d0955204864069f09332aba8fbcaf82fc9127550851baebbe8e5`.

## Scope

The original run's DINOv2 feature cache and fitted stage 4/12/16/17 model
checkpoints remain frozen inputs to the assignment rerun. The checkpoint
provenance and hashes are checked by the archived stage 17 script. This run
does not retrain every historical model stage from scratch.

The current `data/preprocessing_inputs.npz` is a later v3 result, not this
initial v1 bundle. It has 278 changed train assignments and zero changed test
assignments compared with v1. Its later train-only refinement and participant
review decisions are separate stages; this run does not claim to reconstruct
that v3 history from raw pixels alone.

## Artifacts

- `preprocessing_inputs_v1_rebuilt.npz` — byte-identical original v1 bundle.
- `output/named_assignments.csv` — rerun original class assignment.
- `stage1_rerun/` — EDA, canonical image outputs, group IDs, and feature
  outputs generated from the current raw competition dataset.
- `report/` — assignment model provenance and output summary.
- `rebuild_preprocessing_inputs_v1.py` — deterministic v1 bundle assembler.

The main submitted notebook was not modified.
