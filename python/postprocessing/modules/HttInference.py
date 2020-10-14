import ROOT
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
from array import array
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from keras.models import load_model
import os
import tensorflow as tf
class inferencerClass(Module):
    def __init__(self, jetSelection):
        self.jetSel = jetSelection
        self.Nparts = 30
        self.Nsvs   = 5
        base = os.environ['CMSSW_BASE']
        self.model_GRU = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/weights_gru.h5')
        #self.model_GRU = load_model('/uscms/home/jkrupa/nobackup/subjetNN/CMSSW_10_2_11/src/PandaAnalysis/dazsle-tagger/evt/nanofiles/deepJet-v8/v25/weights_gru.h5')
        self.model_IN  = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/weights_IN.h5', custom_objects={'tf': tf})
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
	svs      = Collection(event, "SV")
	met      = Object(event, "MET")
   
        tagger_GRU = np.full(1,-1.,dtype=np.float32)
        tagger_IN  = np.full(1,-1.,dtype=np.float32)

        jet_idx = 0
        min_dphi = 999.
        for ij, jet in enumerate(jets):
            if (jet.pt < 300.): continue
            this_dphi = abs(signedDeltaPhi(met.phi, jet.phi))
            if (this_dphi < min_dphi):
                min_dphi = this_dphi
                jet_idx = ij
        pf_idx = 0

        for ij, jet in enumerate(jets):

            #if jet.pt < 400 or jet.msoftdrop < 30 : continue
            if ( ij < jet_idx ):
               pf_idx = pf_idx + jet.nPFConstituents
               continue
            elif ( ij > jet_idx ): 
               continue
            if jet.nPFConstituents< 1: continue
            ##Fill basic jet properties
            jpt    = jet.pt
            jLSpt    = jet.LSpt
            jeta   = jet.eta
            jphi   = jet.phi
            jmsd   = jet.msoftdrop 
            jLSmsd   = jet.LSmsoftdrop 
            jm   = jet.mass
            jdRLep   = jet.dRLep
            jlsf3   = jet.lsf3
	    jn2b1  = jet.n2b1
	    jLSn2b1  = jet.LSn2b1
	    jdeepTagZqq  = jet.deepTagZqq
	    jdeepTagWqq  = jet.deepTagWqq
	    jn3b1  = jet.n3b1
	    jLSn3b1  = jet.LSn3b1
            try: jtau21 = float(jet.tau2)/float(jet.tau1)
            except: jtau21 = 0.
            try: jtau32 = float(jet.tau3)/float(jet.tau2)
            except: jtau32 = 0.
            try: jtau43 = float(jet.tau4)/float(jet.tau3)
            except: jtau43 = 0.
            try: jLStau21 = float(jet.LStau2)/float(jet.LStau1)
            except: jLStau21 = 0.
            try: jLStau32 = float(jet.LStau3)/float(jet.LStau2)
            except: jLStau32 = 0.
            try: jLStau43 = float(jet.LStau4)/float(jet.LStau3)
            except: jLStau43 = 0.

            jetv.SetPtEtaPhiM(jet.pt, jet.eta, jet.phi, jet.mass)

            ##Fill SV
            svpt   = np.zeros(self.Nsvs, dtype = np.float16)
            svdlen  = np.zeros(self.Nsvs, dtype = np.float16)
            svdlenSig = np.zeros(self.Nsvs, dtype = np.float16)
            svdxy  = np.zeros(self.Nsvs, dtype = np.float16)
            svdxySig = np.zeros(self.Nsvs, dtype = np.float16)
            svchi2 = np.zeros(self.Nsvs, dtype = np.float16)
            svpAngle = np.zeros(self.Nsvs, dtype = np.float16)
            svx = np.zeros(self.Nsvs, dtype = np.float16)
            svy = np.zeros(self.Nsvs, dtype = np.float16)
            svz = np.zeros(self.Nsvs, dtype = np.float16)
            svmass = np.zeros(self.Nsvs, dtype = np.float16)
            svphi = np.zeros(self.Nsvs, dtype = np.float16)
            sveta = np.zeros(self.Nsvs, dtype = np.float16)
            svv = ROOT.TLorentzVector()
            arrIdx = 0
            for isv, sv in enumerate(svs):
                if arrIdx == self.Nsvs: break
                svv.SetPtEtaPhiM(sv.pt, sv.eta, sv.phi, sv.mass)
                if jetv.DeltaR(svv) < 0.8:
                   svpt[arrIdx] = sv.pt/jpt
                   svdlen[arrIdx] = sv.dlen
                   svdlenSig[arrIdx] = sv.dlenSig
                   svdxy[arrIdx] = sv.dxy
                   svdxySig[arrIdx] = sv.dxySig
                   svchi2[arrIdx] = sv.chi2
                   svpAngle[arrIdx] = sv.pAngle
                   svx[arrIdx] = sv.x
                   svy[arrIdx] = sv.y
                   svz[arrIdx] = sv.z
                   sveta[arrIdx] = sv.eta-jeta
                   svphi[arrIdx] = signedDeltaPhi(sv.phi, jphi)
                   svmass[arrIdx] = sv.mass
                   arrIdx += 1      

            ##find candidates associated to jet
            candrange = range(pf_idx, pf_idx + jet.nPFConstituents)

            ##Fill PF candidates
            pfpt          = np.zeros(self.Nparts, dtype = np.float16)
            pfeta         = np.zeros(self.Nparts, dtype = np.float16)
            pfphi         = np.zeros(self.Nparts, dtype = np.float16)
            pftrk         = np.zeros(self.Nparts, dtype = np.float16)
            pfpup         = np.zeros(self.Nparts, dtype = np.float16)
            pfpupnolep    = np.zeros(self.Nparts, dtype = np.float16)
            pfq           = np.zeros(self.Nparts, dtype = np.float16)
            pfid          = np.zeros(self.Nparts, dtype = np.float16)
            pfdz          = np.zeros(self.Nparts, dtype = np.float16)
            pfdxy         = np.zeros(self.Nparts, dtype = np.float16)
            pfdxyerr      = np.zeros(self.Nparts, dtype = np.float16)
            arrIdx = 0
            for ip, part in enumerate(pfcands):
                if ip not in candrange: continue
                if arrIdx == self.Nparts: break
                pfpt[arrIdx]       = part.pt/jpt
                pfeta[arrIdx]      = part.eta - jeta
                pfphi[arrIdx]      = signedDeltaPhi(part.phi, jphi)
                pfpup[arrIdx]      = part.puppiWeight
                pfpupnolep[arrIdx] = part.puppiWeightNoLep
                pfq[arrIdx]        = part.charge
                pfid[arrIdx]       = part.pdgId
                pfdz[arrIdx]       = part.dz
                pfdxy[arrIdx]      = part.d0
                pfdxyerr[arrIdx]   = part.d0Err
                pftrk[arrIdx]      = part.trkChi2 
                arrIdx += 1

            #print(pfpt,pfeta,pfphi,pfdz,pfd0)
            ##define and reshape features
            X = np.vstack([pfpt,pfeta,pfphi,pfdz,pfdxy])
            X = np.reshape(X,(X.shape[0],self.Nparts)).T
            X = np.reshape(X,(1,X.shape[0],X.shape[1]))
 

            tagger_GRU[0] = float(self.model_GRU.predict(X)[0,1])
            tagger_IN[0]  = float(self.model_IN.predict(X)[0,1])
            #assert abs( 1 - float(self.model.predict(X)[0,1]) - float(self.model.predict(X)[0,0])) < 0.02
            #print(X,tagger_GRU[0], tagger_IN[0])
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
 
