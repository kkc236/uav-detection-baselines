# RA-GLGM data-authority correction

Status: preregistration correction made before any learnability, Smoke2,
Screen30, or Formal100 training run.

The authoritative VisDrone train annotations contain 10,345 rows whose
confidence/score field is zero. Two of those rows have zero height and cannot
form valid normalized boxes:

- `0000293_03401_d_0000939.txt:130`: `1008,374,3,0,0,0,0,0`
- `9999999_00590_d_0000267.txt:89`: `545,414,10,0,0,0,0,0`

The conversion and sidecar validator both reject non-positive box dimensions.
The frozen authority therefore distinguishes raw rows from usable sidecars:

- train raw score-zero rows: 10,345
- train invalid zero-area rows excluded: 2
- train valid ignore sidecar boxes: 10,343
- validation raw and valid ignore boxes: 1,410

Positive detection labels remain unchanged at 343,204 train boxes and 38,759
validation boxes. This correction resolves an internal preregistration
contradiction; it does not select or alter any experimental result because no
training gate had started.
