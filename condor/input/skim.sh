#!/bin/bash
set -e
set -x
i=1
TAG=${!i}; i=$((i+1))
DATASET=${!i}; i=$((i+1))
SHORT=${!i}; i=$((i+1))
YEAR=${!i}; i=$((i+1))
ISMC=${!i}; i=$((i+1))
ICHUNK=${!i}; i=$((i+1))
SUBDATE=${!i};i=$((i+1))
CONVERT=${!i};i=$((i+1))
OUTDIR=${!i};i=$((i+1))
CUT="${!i}";i=$((i+1))
FILES="${@:$i}"

TOP=$(pwd)

#arguments = 21Jul20 QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8-hadd QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8-hadd 2017 True 0 nocrab_200721_162730 root://cmsxrootd-site.fnal.gov//store/user/lpcbacon/pancakes/02/2017/UL/hadd//QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8-hadd/nano_mc_2017_0.root

echo ${TAG}
echo ${DATASET}
# release info
lsb_release -r

# Proxy info
# klist -A

source /cvmfs/cms.cern.ch/cmsset_default.sh
#md5sum nanotools*.*z

if [ ${CONVERT} -eq 1 ]; then
       # Convert inputs from EDM to flat nano trees
       printf "%s\n" "${FILES[@]}" > files.txt
       scramv1 project CMSSW CMSSW_10_6_6
       pushd CMSSW_10_6_6/src
       eval `scramv1 runtime -sh`
       popd
       cmsRun nano_from_edm_cfg.py year=${YEAR} inputFiles_load=files.txt
       rm -rf CMSSW_10_2_22
       FILES=( ${TOP}/NanoAOD.root )
fi

# Extract gridpack
tar xf CMSSW*.*z

echo ''
echo '--- Top directory'
ls -l 
echo '-----------------'
echo ''


# Source environment
cd CMSSW_*/src
echo ''
echo '--- BEGIN move'
date
scram b ProjectRename
date
echo '--- END move'
echo ''

eval `scramv1 runtime -sh`


echo ''
echo '--- Src directory'
ls -l 
echo '-----------------'
echo ''

# Go to nanoaod-tools and run


export PYTHONPATH=PYTHONPATH:"${CMSSW_BASE}/lib/${SCRAM_ARCH}"

cd PhysicsTools/NanoAODTools/
echo 'which python: ' $(which python)

echo ''
echo '--- BEGIN Python PATH'
python -c "import sys; print('\n'.join(sys.path))"
echo '--- END Python PATH'
echo ''
echo '--- Starting skim'
echo "${FILES[@]}"
echo "${CUT}"

echo "drop FatJetPFCands_*" >> drop_pfcands.txt

echo "start python $(date)"
python scripts/nano_postproc.py tmp/ ${FILES[@]} -I PhysicsTools.NanoAODTools.postprocessing.modules.HttInference_with_SV inferencer -c "${CUT}" -s "${DATASET}" --bo drop_pfcands.txt
echo "done python $(date)"
#OUTDIR="/store/user/jkrupa/nanopost_process/${TAG}" 
#/$(echo $DATASET | sed "s|/| |g" | awk '{print $1}')/${SHORT}/${SUBDATE}/1337/"
#eos  root://cmseos.fnal.gov/ mkdir -p ${OUTDIR}
#echo "start copying $(date)"
#for i in tmp/*; do xrdcp -f $i root://cmseos.fnal.gov/${OUTDIR}/; done
echo "start hadd $(date)"
#haddnano.py tree_${ICHUNK}.root tmp/*.root
for f in tmp/*.root; do haddnano.py "${f%.root}_skim.root" "${f}"; done
echo "done hadd $(date)"
echo "start copying $(date)"
#xrdcp -f tree_${ICHUNK}.root root://cmseos.fnal.gov/${OUTDIR}/
for i in tmp/*_skim.root; do xrdcp -f $i root://cmseos.fnal.gov/${OUTDIR}/; done
echo "done copying $(date)"
rm -rf tmp/*
