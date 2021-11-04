import datetime
import json
import math
import os
import re
import shutil
import sys

import htcondor

pjoin = os.path.join


def chunkify(items, nchunk):
    '''Split list of items into nchunk ~equal sized chunks'''
    chunks = [[] for _ in range(nchunk)]
    for i in range(len(items)):
        chunks[i % nchunk].append(items[i])
    return chunks

def ismc(dataset):
    return True
    


def setup_dirs(datasets, tag):
    
    # Copy gridpack
    owd = os.getcwd()
    os.chdir(pjoin(os.environ['CMSSW_BASE'],'..'))
    os.system('tar -zvcf CMSSW_11_2_1_%s.tgz CMSSW_11_2_1 --exclude="*.root" --exclude="*.pdf" --exclude="*.pyc" --exclude=tmp --exclude="*.tgz" --exclude-vcs --exclude-caches-all --exclude="*err*" --exclude=*out_* --exclude=*txt --exclude=*jdl'%tag) 
    print("tar complete")
    os.chdir(owd)
   
    for year in datasets:

        # Overall working directory for this tag
        wdir = os.path.abspath("./wdir/%s/%s/"%(tag,year))
        try:
            os.makedirs(wdir)
        except: # FileExistsError:
            pass

        gp_original = pjoin(os.environ['CMSSW_BASE'],'..','CMSSW_11_2_1_%s.tgz'%tag)
        gp = pjoin(wdir, os.path.basename(gp_original))
        shutil.copyfile(gp_original, gp)

        # Loop over datasets
        for item in datasets[year]:
            shortname = item['dataset']
            print('Submitting sample %s'%shortname)
            outdir = "/store/user/lpcbacon/drankin/nanopost_process/%s/%s/%s"%(tag, year, shortname)
    
    
            os.system("eos root://cmseos.fnal.gov/ mkdir -p %s"%outdir) 
            # Working directory for this dataset
            subdir = pjoin(wdir, shortname)
            try:
                os.makedirs(subdir)
            except: #FileExistsError:
                UserWarning ('overwriting files!')
    
            # Submission date identifier
            subdate = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            # Read list of files
            with open(str(item['filelist'])) as f:
                files = [x.strip() for x in f.readlines()]
            cut='(1==1)' #cut = '(FatJet_msoftdrop[0]>30.)&&(FatJet_pt[0]>450.)'
            #if 'JetHT' in shortname: cut += '&&(event%10==0)'
            print('Preparing %i jobs with %i files per job'%(int(math.ceil(len(files) / item['mergefactor'])), item['mergefactor']))
            for ichunk, chunk in enumerate( chunkify(files, int(math.ceil(len(files) / item['mergefactor'])))):
                input_files = [
                    gp,
                    #os.path.abspath('./input/nano_from_edm_cfg.py')
                ]
                arguments = [
                    tag,
                    item['dataset'],
                    shortname,
                    year,
                    ismc(item['dataset']),
                    ichunk,
                    'nocrab_' + subdate,
                    0,# if item['convert'] else 0,
                    outdir,
                    cut,
                ] + chunk
               
    
                # Dfine submission settings
                submission_settings = {
                    "Initialdir" : subdir,
                    "executable": os.path.abspath("./input/skim.sh"),
                    "should_transfer_files" : "YES",
                    "transfer_input_files" : ", ".join(input_files),
                    "arguments": " ".join([str(x) for x in arguments]),
                    "Output" : "out_%s_%s_%i.txt"% ( tag, shortname, ichunk ),
                    "Error" : "err_%s_%s_%i.txt" % ( tag, shortname, ichunk ),
                    "log" : "log_%s_%s_%i.txt"   % ( tag, shortname, ichunk ),
                    "WhenToTransferOutput" : "ON_EXIT_OR_EVICT",
                    "universe" : "vanilla",
                    "request_cpus" : 1,
                    "request_memory" : 2000,
                    #"+MaxRuntime" : "{60*60*8}",
                    "on_exit_remove" : "((ExitBySignal == False) && (ExitCode == 0)) || (NumJobStarts >= 2)",
                    }
    
                # Write JDL file
                jobfile = pjoin(subdir, "skim_%s_%s_%i.jdl"% (tag, shortname, ichunk) )
                sub = htcondor.Submit(submission_settings)
                with open(jobfile,"w") as f:
                    f.write(str(sub))
                    f.write("\nqueue 1\n")
      
                #if ('boostedTau' not in item['dataset'] or ichunk%5==0):
                os.system('condor_submit %s'%jobfile)
                

def main():
    tag = 'Sep14'
    skiplist = ['boostedTau','HTauTau','HToWW','WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8','JetHT']
    whitelist = []
    datasets_dict = {}
    with open("input/fileset_2017.json","r") as f:
        datasets_dict['2017'] = json.loads(f.read())
    with open("input/fileset_2018.json","r") as f:
        datasets_dict['2018'] = json.loads(f.read())
    datasets = {}
    for y in ['2017', '2018']:
        datasets[y] = []
        for d in datasets_dict[y]:
            if any(d['dataset'].startswith(s) for s in skiplist): continue
            if len(whitelist)>0:
                if (not any(w in d['dataset'] for w in whitelist)): continue
            print(d['dataset'])
            datasets[y].append(d)

    setup_dirs(datasets, tag)

if __name__ == "__main__":
    main()
