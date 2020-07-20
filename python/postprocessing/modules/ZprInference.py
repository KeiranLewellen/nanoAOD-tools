import ROOT
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
from array import array
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection 
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from tensorflow.keras.models import load_model

class inferencerClass(Module):
    def __init__(self, jetSelection):
        self.jetSel = jetSelection
        self.Nparts = 20
        self.Nsvs   = 5
        self.model = load_model('/uscms/home/jkrupa/nobackup/subjetNN/CMSSW_10_2_11/src/PandaAnalysis/dazsle-tagger/evt/nanofiles/deepJet-v8/v25/weights_gru.h5')

    def beginJob(self):
        pass
    def endJob(self):
        pass
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        self.out.branch("fj_tagger", "F", 1)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        pfcands	 = Collection(event, "FatJetPFCands")
        jets 	 = Collection(event, "FatJet")
   
        tagger = np.full(1,-1.)
        print('hi')
        for ij, jet in enumerate(jets):
            print(ij)

            if jet.pt < 525 or jet.msoftdrop < 40 : continue

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
            for ip, part in enumerate(pfcands):
                if arrIdx == self.Nparts: break
                if ip not in candrange: continue
                pfpt[arrIdx]  = part.pt*part.puppiWeight/jpt
                pfeta[arrIdx] = part.eta - jeta
                pfphi[arrIdx] = signedDeltaPhi(part.phi, jphi)
                pfdz[arrIdx]  = part.dz
                pfd0[arrIdx]  = part.d0
                arrIdx += 1

            ##define and reshape features
            X = np.vstack([pfpt,pfeta,pfphi,pfdz,pfd0])
            X = np.reshape(X,(X.shape[0],self.Nparts)).T
            print(X)
            X = np.reshape(X,(1,X.shape[0],X.shape[1]))
 
            tagger[ij] = float(self.model.predict(X)[0,0])
            print(str(ij)+' tagger = %.2f'%tagger[ij])   
	self.out.fillBranch("fj_tagger",tagger)
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def signedDeltaPhi(phi1,phi2):
    dPhi = phi1 - phi2
    if(dPhi < -np.pi):
       dPhi = 2*np.pi+dPhi
    elif(dPhi > np.pi):
       dPhi = -2*np.pi+dPhi
    return dPhi

inferencer = lambda : inferencerClass(jetSelection= lambda j : j.pt >= 400. and j.msoftdrop >= 40.) 
 
