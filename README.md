# DB2DPCA

MATLAB experiments for bilateral 2DPCA information fusion in image reconstruction and recognition.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-D4AF37?style=flat-square)](LICENSE)

## Paper

Scripts for *Fusion of Bilateral 2DPCA Information for Image Reconstruction and Recognition*. [Read the paper](https://www.mdpi.com/2076-3417/12/24/12913).

## Prerequisites and Execution

The datasets are external to this repository. Both [classification.m](classification.m) and [reconstruction.m](reconstruction.m) default to `../data/handwritten/BDL`; [database.m](database.m) lists other example locations.

Prepare the data expected by the selected `classify_*.m` or `reco_*.m` function, then edit the `database` variable to the actual path. Classification defaults to `classfier='CRC'` and `nPC=16`. From the repository directory in MATLAB, run the required stage:

```matlab
classification
% Or, for reconstruction:
reconstruction
```

The current checkout does not include the default BDL dataset or a dataset-conversion script.

## Repository Structure

- [classification.m](classification.m) and [reconstruction.m](reconstruction.m): experiment entry points.
- [database.m](database.m): example external input locations.
- [classify_DB2DPCA.m](classify_DB2DPCA.m) and [reco_DB2DPCA.m](reco_DB2DPCA.m): DB2DPCA experiment functions.
- [CRC.m](CRC.m): CRC classifier.

## License

See the existing [GPL-3.0 license](LICENSE).
