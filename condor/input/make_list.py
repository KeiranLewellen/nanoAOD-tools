import os
import subprocess
import json

eosbase = "root://cmseos.fnal.gov/"
eosdir = "/store/group/lpcbacon/pancakes/02/"
dirlist = [
    #["2017/UL/", "2017UL",["Run20","hww_2017mc","hadd","tmp"]],
    ["2017/tmp-VJets-withPF", "2017VJets",[]],#["tmp-VJets-withPF"]],
    #["2018/UL", "2018UL",["200211_180642"]],
    #["2017/UL/JetHT/", "2017ULhadd",["_Run2017B"]]
]

def eos_rec_search(startdir,suffix,skiplist,dirs):
    donedirs = []
    dirlook = subprocess.check_output("eos %s ls %s"%(eosbase,startdir), shell=True).decode('utf-8').split("\n")[:-1]
    print('dirlook', dirlook)
    for d in dirlook:
        if d.endswith(suffix):
            print('file', startdir+"/"+d)
            donedirs.append(startdir+"/"+d)
        elif any(skip in d for skip in skiplist):
            print("Skipping %s"%d)
            continue
        else:
            print("Searching %s"%d)
            donedirs = donedirs + eos_rec_search(startdir+"/"+d,suffix,skiplist,dirs+donedirs)
    #print('dir+donedirs', dirs+donedirs)
    return dirs+donedirs

for dirs in dirlist:
    samples = subprocess.check_output("eos %s ls %s%s"%(eosbase,eosdir,dirs[0]), shell=True).decode('utf-8').split("\n")[:-1]
    print('samples', samples)
    jdict = {}
    for s in samples:
        print("\tRunning on %s"%s)
        curdir = "%s%s/%s"%(eosdir,dirs[0],s)
        dirlog = eos_rec_search(curdir,".root",dirs[2],[])
        dirlog = set(dirlog)
        if not dirlog:
            print("Empty sample skipped")
        else: 
            jdict[s] = [eosbase+d for d in dirlog]
        print(jdict[s])
        outfile = open("%s.txt"%s, "w") 
        #print([fi + "\n" for fi in jdict[s]])
        for fi in jdict[s]: print >>outfile, fi
    #print(dirs[1],[s for s in jdict])
    #with open("fileset%s.json"%(dirs[1]), 'w') as outfile:
    #    json.dump(jdict, outfile, indent=4, sort_keys=True)
