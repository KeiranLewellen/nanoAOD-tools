1. Create inferencer with PFCands/SV/FatJet info (eg. `python/postprocessing/modules/ZprInference.py`).
2. Dump filesets as `.txt` files with `condor/input/make_fileset.py` 
3. Parse fileset/splitting to `.json` with `condor/input/make_json.py`.
4. Job submission from python `condor/skim.py` with condor executable `condor/input/skim.sh` executing main nanoaod-tools postprocess script `scripts/nano_postproc.py`. Add any cuts (eg. blinding on data). 
