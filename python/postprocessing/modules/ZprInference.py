import ROOT
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
from array import array
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection 
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
#from keras.models import load_model
import onnxruntime as rt
import os
#import tensorflow as tf
class inferencerClass(Module):
    def __init__(self, jetSelection):
        self.jetSel = jetSelection
        self.Nparts = 30
        self.Nsvs   = 5
        base = os.environ['CMSSW_BASE']
        #self.model_GRU = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/weights_gru.h5')
        #self.model_GRU = load_model('/uscms/home/jkrupa/nobackup/subjetNN/CMSSW_10_2_11/src/PandaAnalysis/dazsle-tagger/evt/nanofiles/deepJet-v8/v25/weights_gru.h5')
        #self.model_IN  = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/weights_IN_v101.h5', custom_objects={'tf': tf})
        self.model_14Apr21_2016_v1 = {}
        self.model_14Apr21_2016_v1["session"]     = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/14Apr21_preUL_2016_v1.onnx')
        self.model_14Apr21_2016_v1["output_name"] = self.model_14Apr21_2016_v1["session"].get_outputs()[0].name
        self.model_14Apr21_2016_v1["input_name"]  = self.model_14Apr21_2016_v1["session"].get_inputs()[0].name

    def beginJob(self):
        pass
    def endJob(self):
        pass
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        #self.out.branch("GRU_v25", "F", 1)
        #self.out.branch("IN_v3_2017", "F", 1)
        #self.out.branch("IN_14Apr21_2017", "F", 1)
        self.out.branch("IN_14Apr21_2016_v1", "F", 1)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        pfcands	 = Collection(event, "FatJetPFCands")
        jets 	 = Collection(event, "FatJet")
  
        #tagger_GRU_2017    = np.full(1,-1.,dtype=np.float32)
        #tagger_IN_v3_2017  = np.full(1,-1.,dtype=np.float32)
        #tagger_IN_14Apr21_2017 = np.full(1,-1.,dtype=np.float32)
        tagger_IN_14Apr21_2016_v1 = np.full(1,-99.,dtype=np.float32)

        for ij, jet in enumerate(jets):

            ##tagger trained pt > 200 and m > 40
            if not (jet.pt > 200 and jet.msoftdrop > 40) : continue
            if ij>0 : continue
            if jet.nPFConstituents< 1: continue
            ##basic jet properties
            jpt    = jet.pt
            jeta   = jet.eta
            jphi   = jet.phi

            ##prepare candidates 
            pfpt  = np.zeros(self.Nparts, dtype = np.float32)
            pfeta = np.zeros(self.Nparts, dtype = np.float32)
            pfphi = np.zeros(self.Nparts, dtype = np.float32)
            pfdz  = np.zeros(self.Nparts, dtype = np.float32)
            pfd0  = np.zeros(self.Nparts, dtype = np.float32)

            ##find candidates associated to jet
            candrange = range(0, jet.nPFConstituents)

            ##fill normalized to 
            ##nominal features: pt, eta, phi, dz, d0
            arrIdx = 0
            #print(len(pfcands), len(candrange))
            for ip, part in enumerate(pfcands):
                if arrIdx == self.Nparts: break
                if ip not in candrange: continue
                pfpt[arrIdx]  = part.pt/jpt
                pfeta[arrIdx] = part.eta - jeta
                pfphi[arrIdx] = signedDeltaPhi(part.phi, jphi)
                pfdz[arrIdx]  = part.dz
                pfd0[arrIdx]  = part.d0
                arrIdx += 1
            ##define and reshape features
            X = np.vstack([pfpt,pfeta,pfphi,pfdz,pfd0])
            X = np.reshape(X,(X.shape[0],self.Nparts)).T
            X = np.reshape(X,(1,X.shape[0],X.shape[1]))
            X = X.astype(np.float32)
            tagger_IN_14Apr21_2016_v1[ij] = self.model_14Apr21_2016_v1["session"].run([self.model_14Apr21_2016_v1["output_name"]], {self.model_14Apr21_2016_v1["input_name"] : X })[0][0][1]

        self.out.fillBranch("IN_14Apr21_2016_v1",tagger_IN_14Apr21_2016_v1)
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def signedDeltaPhi(phi1,phi2):
    dPhi = phi1 - phi2
    if(dPhi < -np.pi):
       dPhi = 2*np.pi+dPhi
    elif(dPhi > np.pi):
       dPhi = -2*np.pi+dPhi
    return dPhi

inferencer = lambda : inferencerClass(jetSelection= lambda j : j.pt >= 0.)# and j.msoftdrop >= 40.) 
 
