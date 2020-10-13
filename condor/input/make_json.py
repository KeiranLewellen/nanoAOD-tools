import json, glob

year = '2017'

mergefactors = { "JetHT_pancakes-02_Run2017B-09Aug2019_UL2017-v1" : 10,
                 "JetHT_pancakes-02_Run2017C-09Aug2019_UL2017-v1" : 10,
                 "JetHT_pancakes-02_Run2017D-09Aug2019_UL2017-v1" : 10,
                 "JetHT_pancakes-02_Run2017E-09Aug2019_UL2017-v1" : 10,
                 "JetHT_pancakes-02_Run2017F-09Aug2019_UL2017-v1" : 10,
                 "Tau_pancakes-02_Run2017B-09Aug2019_UL2017-v1" : 10,
                 "Tau_pancakes-02_Run2017C-09Aug2019_UL2017-v1" : 10,
                 "Tau_pancakes-02_Run2017D-09Aug2019_UL2017-v1" : 10,
                 "Tau_pancakes-02_Run2017E-09Aug2019_UL2017-v1" : 10,
                 "Tau_pancakes-02_Run2017F-09Aug2019_UL2017-v1" : 10,
                 "SingleElectron_pancakes-02_Run2017B-09Aug2019_UL2017-v1" : 10,
                 "SingleElectron_pancakes-02_Run2017C-09Aug2019_UL2017-v1" : 10,
                 "SingleElectron_pancakes-02_Run2017D-09Aug2019_UL2017-v1" : 10,
                 "SingleElectron_pancakes-02_Run2017E-09Aug2019_UL2017-v1" : 10,
                 "SingleElectron_pancakes-02_Run2017F-09Aug2019_UL2017-v1" : 10,
                 "SingleMuon_pancakes-02-withPF_Run2017B-09Aug2019_UL2017-v1" : 3,
                 "SingleMuon_pancakes-02-withPF_Run2017C-09Aug2019_UL2017-v1" : 3,
                 "SingleMuon_pancakes-02-withPF_Run2017D-09Aug2019_UL2017-v1" : 3,
                 "SingleMuon_pancakes-02-withPF_Run2017E-09Aug2019_UL2017-v1" : 3,
                 "SingleMuon_pancakes-02-withPF_Run2017F-09Aug2019_UL2017-v1" : 3,
                 "DYJetsToLL_Pt-250To400_TuneCP5_13TeV-amcatnloFXFX-pythia8" : 5,
                 "DYJetsToLL_Pt-400To650_TuneCP5_13TeV-amcatnloFXFX-pythia8" : 5,
                 "DYJetsToLL_Pt-650ToInf_TuneCP5_13TeV-amcatnloFXFX-pythia8" : 5,
                 "QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8" : 20,
                 "QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8" : 20,
                 "QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8" : 3,
                 "QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8" : 2,
                 "QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8" : 1,
                 "QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8" : 2,
                 "ZJetsToQQ_HT400to600_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8" : 8, 
                 "ZJetsToQQ_HT600to800_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8" : 8, 
                 "ZJetsToQQ_HT-800toInf_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8" : 8,
                 "WJetsToQQ_HT400to600_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8" : 8, 
                 "WJetsToQQ_HT600to800_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8" : 8, 
                 "WJetsToQQ_HT-800toInf_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8" : 8, 
                 "TTToHadronic_TuneCP5_13TeV-powheg-pythia8" : 5,
                 "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8" : 5, 
                 "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8" : 20,
                 "ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8" : 20,
                 "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8" : 20,
                 "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8" : 20,
                 "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8" : 5,
                 "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8" : 5,
                 "ST_tW_top_5f_inclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8" : 5,
                 "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8" : 2,
                 "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8" : 5, 
                 "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8" : 5, 
                 "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8" : 5,
                 "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8" : 5,  
                 "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8" : 5,  
                 "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8" : 5,  
                 "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8" : 5,  
                 "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8" : 5,  
                } 
of = open('fileset_%s.json'%year,'w')
of.write('[\n')
for i, s in enumerate(glob.glob(year+'/*')):


    of.write('\n\t{')
    of.write('"dataset" : "%s",\n'%s.replace('.txt','').replace(year+'/',''))
    of.write('\t"filelist" : "input/%s",\n'%s)

    of.write('\t"mergefactor" : %i\n\t},'%mergefactors[s.replace('.txt','').replace(year+'/','')])
of.write('\n]')
of.close()
