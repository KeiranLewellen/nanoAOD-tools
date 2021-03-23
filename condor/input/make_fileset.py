import os
import subprocess
import json

eosbase = "root://cmseos.fnal.gov/"
eosdir = "/store/group/lpcbacon/pancakes/02/"

dirlist = [
    #["2017/UL/", "2017",["hadd","MET","DYJets","GluGlu","Electron","WJetsToLNu","WW","WZ","ZZ","JetHT/pancakes-02_Run2017B", "QCD_HT100to200", "QCD_HT200to300","QCD_HT50to100","SingleMuon","JetHT","SingleElectron","Tau"],{"SingleMuon":"withPF",},],
    ["2017/UL/", "2017",["hadd","MET","DYJets","GluGlu","Electron","WJetsToLNu","WW","WZ","ZZ","JetHT/pancakes-02_Run2017B", "QCD_HT100to200", "QCD_HT200to300","QCD_HT50to100","SingleMuon","JetHT","SingleElectron","Tau"],{"SingleMuon":"withPF","SingleElectron":"withPF"},],
    ["2017/", "2017", ["UL","tmp","JetHT","Single","ZPrime","BulkGrav"],{"GluGluHToTauTau":"TauTau","VBFHToTauTau":"TauTau","WminusHToTauTau":"TauTau","WplusHToTauTau":"TauTau","ZHToTauTau":"TauTau","ggZH_HToTauTau_ZToLL":"TauTau","ggZH_HToTauTau_ZToNuNu":"TauTau","ggZH_HToTauTau_ZToQQ":"TauTau","ttHToTauTau":"TauTau"}],
    ["2017/tmp-VJets-withPF", "2017",["UL"], {},],#["tmp-VJets-withPF"]],
    ["2017/tmp-VJets-withPF/UL", "2017",[], {},],#["tmp-VJets-withPF"]],
    ["2017/tmp-WJets-withPF", "2017",[], {},],#["tmp-VJets-withPF"]],
    ["2017/UL/SingleMuon","2017",[],{"SingleMuon":"withPF"}],
    ["2017/UL/SingleElectron","2017",[],{"SingleElectron":"withPF"}],
    ["2017/UL/MET","2017",[],{"MET":"withPF"}],
    ["2017/UL/JetHT","2017",[],{}], 
    #["2017/UL/Tau","2017",[],{}], 
    #["2017/UL/SingleElectron","2017",[],{}], 
    #["2018/UL", "2018UL",["200211_180642"]],
    #["2017/UL/hadd", "2017ULhadd",["_Run2017B"]]
    #["","2017",[]]   
    ["2018/UL/GluGluHTauTau_13TeV_user", "2018",[], {},],
]

def eos_rec_search(startdir,suffix,skiplist,dirs,reqs=""):
    donedirs = []
    dirlook = subprocess.check_output("eos %s ls %s"%(eosbase,startdir), shell=True).decode('utf-8').split("\n")[:-1]
    print(reqs,startdir)
    if reqs not in startdir: return []#os.system("eos %s ls %s"%(eosbase,startdir))): return []
    for d in dirlook:
        print(reqs,d)
        if d.endswith(suffix):
            #print('file', startdir+"/"+d)
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


        if any(skip in s for skip in dirs[2]): continue #print('skipping',s);  continue
        print("\tRunning on %s"%s)
        curdir = "%s%s/%s/"%(eosdir,dirs[0],s)
        name = s.replace('crab_','').replace('_user','')
        if 'SingleMuon' in curdir: name = 'SingleMuon_' + name
        if 'JetHT' in curdir: name = 'JetHT_' + name
        if 'Tau' in curdir and 'TauTau' not in curdir: name = 'Tau_' + name
        if 'SingleElectron' in curdir: name = 'SingleElectron_' + name
        if 'MET' in curdir: name = 'MET_' + name
        

        requirements = ""
        for k,v in dirs[3].items():
            if k in name: requirements = v
        if requirements: print('path for sample %s must contain %s'%(name, requirements))
        print(curdir)
        #try: 
         
        #    requirements = dirs[3][s]
        #    print('path for sample %s must contain %s'%(s, requirements))
        #except:
        #    requirements = ""
        #    pass #continue
        dirlog = eos_rec_search(curdir,".root",dirs[2],[], requirements)
        dirlog = set(dirlog)
        if not dirlog:
            print("Empty sample skipped")
            continue
        else: 
            jdict[s] = [eosbase+d for d in dirlog]
        #print(jdict[s])

        os.system('mkdir -p %s'%dirs[1])
        out = open('%s/%s.txt'%(dirs[1], name),'w')
        for f in jdict[s]:
             #if not any(skip in f for skip in dirs[2]):
             out.write(f)
             out.write('\n')
        out.close()
    #with open("fileset%s.json"%(dirs[1]), 'w') as outfile:
    #    json.dump(jdict, outfile, indent=4, sort_keys=True)


