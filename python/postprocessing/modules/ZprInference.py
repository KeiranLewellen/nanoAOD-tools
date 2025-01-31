import ROOT
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
from array import array
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection 
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from keras.models import load_model
import os
import tensorflow as tf
class inferencerClass(Module):
    def __init__(self, jetSelection):
        self.jetSel = jetSelection
        self.Nparts = 20
        self.Nsvs   = 5
        base = os.environ['CMSSW_BASE']
        self.model_GRU = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/weights_gru.h5')
        #self.model_GRU = load_model('/uscms/home/jkrupa/nobackup/subjetNN/CMSSW_10_2_11/src/PandaAnalysis/dazsle-tagger/evt/nanofiles/deepJet-v8/v25/weights_gru.h5')
        self.model_IN  = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/weights_IN_v101.h5', custom_objects={'tf': tf})
    def beginJob(self):
        pass
    def endJob(self):
        pass
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        self.out.branch("GRU_v25", "F", 1)
        self.out.branch("IN_v3", "F", 1)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        pfcands	 = Collection(event, "FatJetPFCands")
        jets 	 = Collection(event, "FatJet")
   
        tagger_GRU = np.full(1,-1.,dtype=np.float32)
        tagger_IN  = np.full(1,-1.,dtype=np.float32)
        for ij, jet in enumerate(jets):

            #if jet.pt < 400 or jet.msoftdrop < 30 : continue
            if ij>0 : continue
            if jet.nPFConstituents< 1: continue
            ##basic jet properties
            jpt    = jet.pt
            jeta   = jet.eta
            jphi   = jet.phi

            ##prepare candidates 
            pfpt  = np.zeros(self.Nparts, dtype = np.float16)
            pfeta = np.zeros(self.Nparts, dtype = np.float16)
            pfphi = np.zeros(self.Nparts, dtype = np.float16)
            pfdz  = np.zeros(self.Nparts, dtype = np.float16)
            pfd0  = np.zeros(self.Nparts, dtype = np.float16)

            ##find candidates associated to jet
            candrange = range(0, jet.nPFConstituents)

            ##fill normalized to 
            ##nominal features: pt, eta, phi, dz, d0
            arrIdx = 0
            #print(len(pfcands), len(candrange))
            for ip, part in enumerate(pfcands):
                if arrIdx == self.Nparts: break
                if ip not in candrange: continue
                pfpt[arrIdx]  = part.pt*part.puppiWeight/jpt
                pfeta[arrIdx] = part.eta - jeta
                pfphi[arrIdx] = signedDeltaPhi(part.phi, jphi)
                pfdz[arrIdx]  = part.dz
                pfd0[arrIdx]  = part.d0
                arrIdx += 1
            #print(pfpt,pfeta,pfphi,pfdz,pfd0)
            ##define and reshape features
            X = np.vstack([pfpt,pfeta,pfphi,pfdz,pfd0])
            X = np.reshape(X,(X.shape[0],self.Nparts)).T
            X = np.reshape(X,(1,X.shape[0],X.shape[1]))
 

            tagger_GRU[ij] = float(self.model_GRU.predict(X)[0,1])
            tagger_IN[ij]  = float(self.model_IN.predict(X)[0,1])
            #assert abs( 1 - float(self.model.predict(X)[0,1]) - float(self.model.predict(X)[0,0])) < 0.02
            #print(X,tagger_GRU[ij], tagger_IN[ij])
	self.out.fillBranch("GRU_v25",tagger_GRU)
	self.out.fillBranch("IN_v3",tagger_IN)
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
 
