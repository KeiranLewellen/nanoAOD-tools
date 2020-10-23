1. Create inferencer with PFCands/SV/FatJet info (eg. `python/postprocessing/modules/HttInference.py`).
2. Dump filesets as `.txt` files with `condor/input/make_fileset.py` 
3. Parse fileset/splitting to `.json` with `condor/input/make_json.py`.
4. Job submission from python `condor/skim.py` with condor executable `condor/input/skim.sh` executing main nanoaod-tools postprocess script `scripts/nano_postproc.py`. Add any cuts (eg. blinding on data). 
5. Alternatively run locally with ex. `python scripts/nano_postproc.py testing/ root://cmseos.fnal.gov//store/group/lpcbacon/pancakes/02/2018/UL/GluGluHTauTau_13TeV_user/crab_GluGluHTauTau_13TeV_user//200222_172425/0000/nano_mc_2018_874.root -I PhysicsTools.NanoAODTools.postprocessing.modules.HttInference inferencer`
