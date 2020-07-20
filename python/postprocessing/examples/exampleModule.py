import ROOT
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
from array import array
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection 
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
SVDRMATCH = 0.8
GENDRMATCH = 0.6
BpdgId = 5
CpdgId = 4
class pfcandProducer(Module):
    def __init__(self, jetSelection):
        self.jetSel = jetSelection
        self.Nparts = 30
        self.Nsvs   = 5
    def beginJob(self):
        pass
    def endJob(self):
        pass
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        self.out.branch("fj_idx", "I", 1);
        self.out.branch("fj_msd", "F", 1);
        self.out.branch("fj_m", "F", 1);
	self.out.branch("fj_pt", "F", 1);
	self.out.branch("fj_eta", "F", 1);
 	self.out.branch("fj_phi", "F", 1);
 	self.out.branch("fj_n2b1", "F", 1);
 	self.out.branch("fj_tau21", "F", 1);
 	self.out.branch("fj_deepTagZqq", "F", 1);
 	self.out.branch("fj_deepTagWqq", "F", 1);
	self.out.branch("fj_nPFCands","I",1);
	self.out.branch("fj_nProngs","I",1);
	self.out.branch("fj_nBtags","I",1);
	self.out.branch("fj_nCtags","I",1);


        self.out.branch("PF_pt",  "F", self.Nparts);
        self.out.branch("PF_eta",  "F", self.Nparts);
        self.out.branch("PF_phi",  "F", self.Nparts);
        self.out.branch("PF_pup",  "F", self.Nparts);
        self.out.branch("PF_q",  "F", self.Nparts);
        self.out.branch("PF_trk",  "F", self.Nparts);
        self.out.branch("PF_dz",  "F", self.Nparts);
        self.out.branch("PF_d0",  "F", self.Nparts);
 
        self.out.branch("SV_dlen","F", self.Nsvs)
        self.out.branch("SV_dlenSig","F", self.Nsvs)
        self.out.branch("SV_dxy","F", self.Nsvs)
        self.out.branch("SV_dxySig","F", self.Nsvs)
        self.out.branch("SV_chi2","F", self.Nsvs)
        self.out.branch("SV_pAngle","F", self.Nsvs)
        self.out.branch("SV_x","F", self.Nsvs)
        self.out.branch("SV_y","F", self.Nsvs)
        self.out.branch("SV_z","F", self.Nsvs)
        self.out.branch("SV_pt","F", self.Nsvs)
        self.out.branch("SV_mass","F", self.Nsvs)
        self.out.branch("SV_eta","F", self.Nsvs)
        self.out.branch("SV_phi","F", self.Nsvs)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        pfcands	 = Collection(event, "FatJetPFCands")
        jets 	 = Collection(event, "FatJet")
	genparts = Collection(event, "GenPart")
	SV       = Collection(event, "SV")
        for ij, jet in enumerate(jets):
 

            if jet.pt < 400 or jet.msoftdrop < 40 : continue

            ##Fill basic jet properties
            jpt    = jet.pt
            jeta   = jet.eta
            jphi   = jet.phi
            jmsd   = jet.msoftdrop 
            jm   = jet.mass
	    jn2b1  = jet.n2b1
	    jdeepTagZqq  = jet.deepTagZqq
	    jdeepTagWqq  = jet.deepTagWqq
	    jn2b1  = jet.n2b1
            try: jtau21 = float(jet.tau2)/float(jet.tau1)
            except: jtau21 = -1.



            ##Calculate # prongs by looping over daughters of Z'
            status = 0
            motheridx = None
            jetv  = ROOT.TLorentzVector()
            jetv.SetPtEtaPhiM(jpt, jeta, jphi, jmsd) 
            dau1 = ROOT.TLorentzVector()
            dau2 = ROOT.TLorentzVector()
	    for ig, gpart in enumerate(genparts):
                if gpart.pdgId == 55 and gpart.status > status:
                    motheridx = ig
                    status = gpart.status

            jdau1pdgId = 0
            jdau2pdgId = 0
	    for ig, gpart in enumerate(genparts):
                if type(motheridx) == None: break
                if gpart.genPartIdxMother == motheridx:
                    if dau1.Pt() == 0.: 
                      dau1.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                      jdau1pdgId = gpart.pdgId
                    elif dau2.Pt() == 0.: 
                      dau2.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                      jdau2pdgId = gpart.pdgId

            ##Flavour and prong tags
            nBtags = 0
            if abs(jdau1pdgId) == BpdgId:
                nBtags += 1
            if abs(jdau2pdgId) == BpdgId:
                nBtags += 1
            nCtags = 0
            if abs(jdau1pdgId) == CpdgId:
                nCtags += 1
            if abs(jdau2pdgId) == CpdgId:
                nCtags += 1
            nProngs = 0
            if dau1.Pt() > 0. and dau2.Pt() > 0.:
                if jetv.DeltaR(dau1) < GENDRMATCH: nProngs += 1
                if jetv.DeltaR(dau2) < GENDRMATCH: nProngs += 1

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
            for isv, sv in enumerate(SV):
                if arrIdx == self.Nsvs: break
                svv.SetPtEtaPhiM(sv.pt, sv.eta, sv.phi, sv.mass)
                if jetv.DeltaR(svv) < SVDRMATCH:
                   svpt[arrIdx] = sv.pt
                   svdlen[arrIdx] = sv.dlen
                   svdlenSig[arrIdx] = sv.dlenSig
                   svdxy[arrIdx] = sv.dxy
                   svdxySig[arrIdx] = sv.dxySig
                   svchi2[arrIdx] = sv.chi2
                   svpAngle[arrIdx] = sv.pAngle
                   svx[arrIdx] = sv.x
                   svy[arrIdx] = sv.y
                   svz[arrIdx] = sv.z
                   sveta[arrIdx] = sv.eta
                   svphi[arrIdx] = sv.phi
                   svmass[arrIdx] = sv.mass
                   arrIdx += 1      

            ##Fill PF candidates
            jnPFCands = jet.nPFConstituents
            if ij == 0:  candrange = range(0, jet.nPFConstituents)
            elif ij > 0: candrange = range(jets[ij-1].nPFConstituents, jets[ij-1].nPFConstituents + jets[ij].nPFConstituents)
            if candrange == []: print jets[ij-1].nPFConstituents, jets[ij].nPFConstituents, jet.pt,jet.msoftdrop
            pfpt  = np.zeros(self.Nparts, dtype = np.float16)
            pfeta = np.zeros(self.Nparts, dtype = np.float16)
            pfphi = np.zeros(self.Nparts, dtype = np.float16)
            pftrk = np.zeros(self.Nparts, dtype = np.float16)
            pfpup = np.zeros(self.Nparts, dtype = np.float16)
            pfq   = np.zeros(self.Nparts, dtype = np.float16)
            pfdz  = np.zeros(self.Nparts, dtype = np.float16)
            pfd0  = np.zeros(self.Nparts, dtype = np.float16)
            arrIdx = 0

            for ip, part in enumerate(pfcands): 
                if arrIdx == self.Nparts: break
                if ip not in candrange: continue
                pfpt[arrIdx]  = part.pt*part.puppiWeight/jpt
                pfeta[arrIdx] = part.eta - jeta
                pfphi[arrIdx] = signedDeltaPhi(part.phi, jphi)
                pfpup[arrIdx] = part.puppiWeight
                pfq[arrIdx]   = part.charge
                pfdz[arrIdx]  = part.dz
                pfd0[arrIdx]  = part.d0
                pftrk[arrIdx] = part.trkChi2 
                arrIdx += 1


	    self.out.fillBranch("fj_idx",ij)
	    self.out.fillBranch("fj_pt",jpt)
	    self.out.fillBranch("fj_eta",jeta)
            self.out.fillBranch("fj_phi",jphi)
	    self.out.fillBranch("fj_n2b1",jn2b1)
	    self.out.fillBranch("fj_tau21",jtau21)
	    self.out.fillBranch("fj_nProngs",nProngs)
	    self.out.fillBranch("fj_nBtags",nBtags)
	    self.out.fillBranch("fj_nCtags",nCtags)
	    self.out.fillBranch("fj_nPFCands",jnPFCands)
	    self.out.fillBranch("fj_deepTagZqq",jdeepTagZqq)
	    self.out.fillBranch("fj_deepTagWqq",jdeepTagWqq)
	    self.out.fillBranch("fj_msd",jmsd)
	    self.out.fillBranch("fj_m",jm)
            self.out.fillBranch("PF_pt",pfpt)
            self.out.fillBranch("PF_eta",pfeta)
            self.out.fillBranch("PF_phi",pfphi)
            self.out.fillBranch("PF_pup",pfpup)
            self.out.fillBranch("PF_q",pfq)
            self.out.fillBranch("PF_dz",pfdz)
            self.out.fillBranch("PF_d0",pfd0)
            self.out.fillBranch("PF_trk",pftrk)
            self.out.fillBranch("SV_dlen", svdlen)
            self.out.fillBranch("SV_dlenSig", svdlenSig)
            self.out.fillBranch("SV_dxy", svdxy)
            self.out.fillBranch("SV_dxySig", svdxySig)
            self.out.fillBranch("SV_chi2", svchi2)
            self.out.fillBranch("SV_pAngle", svpAngle)
            self.out.fillBranch("SV_x", svx)
            self.out.fillBranch("SV_y", svy)
            self.out.fillBranch("SV_z", svz)
            self.out.fillBranch("SV_pt", svpt)
            self.out.fillBranch("SV_mass", svmass)
            self.out.fillBranch("SV_eta", sveta)
            self.out.fillBranch("SV_phi", svphi)
            return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def signedDeltaPhi(phi1,phi2):
    dPhi = phi1 - phi2
    if(dPhi < -np.pi):
       dPhi = 2*np.pi+dPhi
    elif(dPhi > np.pi):
       dPhi = -2*np.pi+dPhi
    return dPhi

pfModule = lambda : pfcandProducer(jetSelection= lambda j : j.pt >= 400. and j.msoftdrop >= 40.) 
 
