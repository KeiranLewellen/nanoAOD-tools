
import ROOT
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
from array import array
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from keras.models import load_model
import json
import os
import tensorflow as tf

class inferencerClass(Module):
    def __init__(self, jetSelection):
        self.jetSel = jetSelection
        self.Nparts = 30
        self.Nsvs = 5
        self.Ntaus = 3
        base = os.environ['CMSSW_BASE']

        #Defines the interaction matrices

        #Defines the recieving matrix for particles
        RR=[]
        for i in range(self.Nparts):
            row=[]
            for j in range(self.Nparts*(self.Nparts-1)):
                if j in range(i*(self.Nparts-1),(i+1)*(self.Nparts-1)):
                    row.append(1.0)
                else:
                    row.append(0.0)
            RR.append(row)
        RR=np.array(RR)
        RR=np.float32(RR)
        RRT=np.transpose(RR)

        #Defines the sending matrix for particles
        RST=[]
        for i in range(self.Nparts):
            for j in range(self.Nparts):
                row=[]
                for k in range(self.Nparts):
                    if k == j:
                        row.append(1.0)
                    else:
                        row.append(0.0)
                RST.append(row)
        rowsToRemove=[]
        for i in range(self.Nparts):
            rowsToRemove.append(i*(self.Nparts+1))
        RST=np.array(RST)
        RST=np.float32(RST)
        RST=np.delete(RST,rowsToRemove,0)
        RS=np.transpose(RST)

        #Defines the recieving matrix for the bipartite particle and secondary vertex graph
        RK=[]
        for i in range(self.Nparts):
            row=[]
            for j in range(self.Nparts*self.Nsvs):
                if j in range(i*self.Nsvs,(i+1)*self.Nsvs):
                    row.append(1.0)
                else:
                    row.append(0.0)
            RK.append(row)
        RK=np.array(RK)
        RK=np.float32(RK)
        RKT=np.transpose(RK)

        #Defines the sending matrix for the bipartite particle and secondary vertex graph
        RV=[]
        for i in range(self.Nsvs):
            row=[]
            for j in range(self.Nparts*self.Nsvs):
                if j % self.Nsvs == i:
                    row.append(1.0)
                else:
                    row.append(0.0)
            RV.append(row)
        RV=np.array(RV)
        RV=np.float32(RV)
        RVT=np.transpose(RV)

        #self.model4p1_hadhad_old = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/IN_hadhad_v4p1,on_QCD,fillFactor=2,200GeV,take_3,model.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
        #self.model6p1_hadel_old = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/GRU_hadel_v6p1,on_TTbar_WJets,fillFactor=1_5,200GeV,ohe,take_8,model.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
        #self.model6p1_hadmu_old = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/GRU_hadmu_v6p1,on_TTbar_WJets,fillFactor=1_5,200GeV,ohe,take_8,model.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})

        self.model4p1_hadhad = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/IN_hadhad_v4p1,on_QCD,fillFactor=2,200GeV,ohe,take_5,model.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
        self.model4p1_hadel = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/IN_hadel_v4p1,on_TTbar_WJets,fillFactor=1_5,200GeV,ohe,take_1,model.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
        self.model4p1_hadmu = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/IN_hadmu_v4p1,on_TTbar_WJets,fillFactor=1_5,200GeV,ohe,take_1,model.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})

        self.postTagger1p1_hadhad = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/PostTagger_hadhad_v1p1,on_QCD,take_1,model.h5')
        self.postTagger1p1_hadel = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/PostTagger_hadel_v1p1,on_TTbar_WJets,take_1,model.h5')
        self.postTagger1p1_hadmu = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/PostTagger_hadmu_v1p1,on_TTbar_WJets,take_1,model.h5')

        self.Ztagger = load_model(base+'/src/PhysicsTools/NanoAODTools/data/GRU_ZDecays_v6p1,5_decays,take_3,model.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})

        #print(self.model4p1_hadhad.summary())
        #print(self.model6p1_hadel.summary())
        #print(self.model6p1_hadmu.summary())

#        json_file_hadhad = open(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadhad_regression_v0.json', 'r')
#        model_json_hadhad = json_file_hadhad.read()
#        self.massreg_hadhad = model_from_json(model_json_hadhad)
#        self.massreg_hadhad.load_weights(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadhad_regression_v0_weights.h5')
#
#        json_file_hadel = open(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadel_regression_v0.json', 'r')
#        model_json_hadel = json_file_hadel.read()
#        self.massreg_hadel = model_from_json(model_json_hadel)
#        self.massreg_hadel.load_weights(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadel_regression_v0_weights.h5')
#
#        json_file_hadmu = open(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadmu_regression_v0.json', 'r')
#        model_json_hadmu = json_file_hadmu.read()
#        self.massreg_hadmu = model_from_json(model_json_hadmu)
#        self.massreg_hadmu.load_weights(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadmu_regression_v0_weights.h5')

        #self.massreg_hadhad = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadhad_regression_v0.h5')
        #self.massreg_hadel = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadel_regression_v0.h5')
        #self.massreg_hadmu = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/fullmodel_hadmu_regression_v0.h5')
        self.massreg_hadhad = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/mass_and_pt_regression_hadhad_v1.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
        self.massreg_hadel = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/mass_and_pt_regression_hadel_v1.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})
        self.massreg_hadmu = load_model(base+ '/src/PhysicsTools/NanoAODTools/data/mass_and_pt_regression_hadmu_v1.h5',custom_objects={'tf': tf,'RK': RK,'RV': RV,'RS': RS,'RR': RR,'RRT': RRT,'RKT': RKT})

        #self.massreg_hadhad = load_model(base+ '/src/TrainMassReg/models/fullmodel_hadhad_regression_001_norm_mae_add13k.h5',custom_objects={'tf':tf})
        #self.massreg_hadel = load_model(base+ '/src/TrainMassReg/models/fullmodel_hadel_regression_v0.h5',custom_objects={'tf':tf})
        #self.massreg_hadmu = load_model(base+ '/src/TrainMassReg/models/fullmodel_hadmu_regression_v0.h5',custom_objects={'tf':tf})

        #print('HADHAD')
        #print(self.massreg_hadhad.summary())
        #print(self.massreg_hadhad.get_weights())
        #print('HADEL')
        #print(self.massreg_hadel.summary())
        #print(self.massreg_hadel.get_weights())
        #print('HADMU')
        #print(self.massreg_hadmu.summary())
        #print(self.massreg_hadmu.get_weights())

        self.log_pf = []
        self.log_sv = []
        self.log_evt = []
        self.log_mreg = []

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree


        self.out.branch("IN_hadhad_v4p1", "F", 1)
        self.out.branch("IN_hadel_v4p1", "F", 1)
        self.out.branch("IN_hadmu_v4p1", "F", 1)

        self.out.branch("PostTagger_hadhad_v1p1", "F", 1)
        self.out.branch("PostTagger_hadel_v1p1", "F", 1)
        self.out.branch("PostTagger_hadmu_v1p1", "F", 1)

        self.out.branch("Ztagger_Zee", "F", 1)
        self.out.branch("Ztagger_Zmm", "F", 1)
        self.out.branch("Ztagger_Zhh", "F", 1)
        self.out.branch("Ztagger_Zhe", "F", 1)
        self.out.branch("Ztagger_Zhm", "F", 1)

        self.out.branch("MassReg_hadhad_mass", "F", 1)
        self.out.branch("MassReg_hadhad_pt", "F", 1)
        self.out.branch("MassReg_hadel_mass", "F", 1)
        self.out.branch("MassReg_hadel_pt", "F", 1)
        self.out.branch("MassReg_hadmu_mass", "F", 1)
        self.out.branch("MassReg_hadmu_pt", "F", 1)


    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        pfcands = Collection(event, "FatJetPFCands")
        jets = Collection(event, "FatJet")
        svs = Collection(event, "SV")
        taus = Collection(event, "Tau")
        met = Object(event, "MET")
        pupmet = Object(event, "PuppiMET")


        IN_hadhad_v4p1 = np.full(1, -1., dtype=np.float32)
        IN_hadel_v4p1 = np.full(1, -1., dtype=np.float32)
        IN_hadmu_v4p1 = np.full(1, -1., dtype=np.float32)

        PostTagger_hadhad_v1p1 = np.full(1, -1., dtype=np.float32)
        PostTagger_hadel_v1p1 = np.full(1, -1., dtype=np.float32)
        PostTagger_hadmu_v1p1 = np.full(1, -1., dtype=np.float32)

        Ztagger_Zee = np.full(1, -1., dtype=np.float32)
        Ztagger_Zmm = np.full(1, -1., dtype=np.float32)
        Ztagger_Zhh = np.full(1, -1., dtype=np.float32)
        Ztagger_Zhe = np.full(1, -1., dtype=np.float32)
        Ztagger_Zhm = np.full(1, -1., dtype=np.float32)

        MassReg_hadhad_mass = np.full(1, -1., dtype=np.float32)
        MassReg_hadhad_pt   = np.full(1, -1., dtype=np.float32)
        MassReg_hadel_mass = np.full(1, -1., dtype=np.float32)
        MassReg_hadel_pt = np.full(1, -1., dtype=np.float32)
        MassReg_hadmu_mass = np.full(1, -1., dtype=np.float32)
        MassReg_hadmu_pt = np.full(1, -1., dtype=np.float32)

        jet_idx = -1
        min_dphi = 999.
        for ij, jet in enumerate(jets):
            if (jet.pt < 200.): continue
            this_dphi = abs(signedDeltaPhi(met.phi, jet.phi))
            if (this_dphi < min_dphi):
                min_dphi = this_dphi
                jet_idx = ij
        pf_idx = 0

        for ij, jet in enumerate(jets):

            # if jet.pt < 400 or jet.msoftdrop < 30 : continue
            if (ij < jet_idx):
                pf_idx = pf_idx + jet.nPFConstituents
                continue
            elif (ij > jet_idx):
                continue
            if jet.nPFConstituents < 1: continue
            ##Fill basic jet properties
            jpt = jet.pt
            jLSpt = jet.LSpt
            jeta = jet.eta
            jphi = jet.phi
            jmsd = jet.msoftdrop
            jLSmsd = jet.LSmsoftdrop
            jm = jet.mass
            jdRLep = jet.dRLep
            jlsf3 = jet.lsf3
            jn2b1 = jet.n2b1
            jLSn2b1 = jet.LSn2b1
            jdeepTagZqq = jet.deepTagZqq
            jdeepTagWqq = jet.deepTagWqq
            jn3b1 = jet.n3b1
            jLSn3b1 = jet.LSn3b1
            try:
                jtau21 = float(jet.tau2) / float(jet.tau1)
            except:
                jtau21 = 0.
            try:
                jtau32 = float(jet.tau3) / float(jet.tau2)
            except:
                jtau32 = 0.
            try:
                jtau43 = float(jet.tau4) / float(jet.tau3)
            except:
                jtau43 = 0.
            try:
                jLStau21 = float(jet.LStau2) / float(jet.LStau1)
            except:
                jLStau21 = 0.
            try:
                jLStau32 = float(jet.LStau3) / float(jet.LStau2)
            except:
                jLStau32 = 0.
            try:
                jLStau43 = float(jet.LStau4) / float(jet.LStau3)
            except:
                jLStau43 = 0.

            jetv = ROOT.TLorentzVector()
            jetv.SetPtEtaPhiM(jet.pt, jet.eta, jet.phi, jet.mass)

            ##Fill SV
            svpt = np.zeros(self.Nsvs, dtype=np.float16)
            svdlen = np.zeros(self.Nsvs, dtype=np.float16)
            svdlenSig = np.zeros(self.Nsvs, dtype=np.float16)
            svdxy = np.zeros(self.Nsvs, dtype=np.float16)
            svdxySig = np.zeros(self.Nsvs, dtype=np.float16)
            svchi2 = np.zeros(self.Nsvs, dtype=np.float16)
            svpAngle = np.zeros(self.Nsvs, dtype=np.float16)
            svx = np.zeros(self.Nsvs, dtype=np.float16)
            svy = np.zeros(self.Nsvs, dtype=np.float16)
            svz = np.zeros(self.Nsvs, dtype=np.float16)
            svmass = np.zeros(self.Nsvs, dtype=np.float16)
            svphi = np.zeros(self.Nsvs, dtype=np.float16)
            sveta = np.zeros(self.Nsvs, dtype=np.float16)
            svv = ROOT.TLorentzVector()
            arrIdx = 0
            for isv, sv in enumerate(svs):
                if arrIdx == self.Nsvs: break
                svv.SetPtEtaPhiM(sv.pt, sv.eta, sv.phi, sv.mass)
                if jetv.DeltaR(svv) < 0.8:
                    svpt[arrIdx] = sv.pt / jpt
                    svdlen[arrIdx] = sv.dlen
                    svdlenSig[arrIdx] = sv.dlenSig
                    svdxy[arrIdx] = sv.dxy
                    svdxySig[arrIdx] = sv.dxySig
                    svchi2[arrIdx] = sv.chi2
                    svpAngle[arrIdx] = sv.pAngle
                    svx[arrIdx] = sv.x
                    svy[arrIdx] = sv.y
                    svz[arrIdx] = sv.z
                    sveta[arrIdx] = sv.eta - jeta
                    svphi[arrIdx] = signedDeltaPhi(sv.phi, jphi)
                    svmass[arrIdx] = sv.mass
                    arrIdx += 1

            # Fill Taus
            tau_charge = np.zeros(self.Ntaus, dtype=np.float16)
            tau_chargedIso = np.zeros(self.Ntaus, dtype=np.float16)
            tau_dxy = np.zeros(self.Ntaus, dtype=np.float16)
            tau_dz = np.zeros(self.Ntaus, dtype=np.float16)
            tau_eta = np.zeros(self.Ntaus, dtype=np.float16)
            tau_leadTkDeltaEta = np.zeros(self.Ntaus, dtype=np.float16)
            tau_leadTkDeltaPhi = np.zeros(self.Ntaus, dtype=np.float16)
            tau_leadTkPtOverTauPt = np.zeros(self.Ntaus, dtype=np.float16)
            tau_mass = np.zeros(self.Ntaus, dtype=np.float16)
            tau_neutralIso = np.zeros(self.Ntaus, dtype=np.float16)
            tau_phi = np.zeros(self.Ntaus, dtype=np.float16)
            tau_photonsOutsideSignalCone = np.zeros(self.Ntaus, dtype=np.float16)
            tau_pt = np.zeros(self.Ntaus, dtype=np.float16)
            tau_rawAntiEle = np.zeros(self.Ntaus, dtype=np.float16)
            tau_rawIso = np.zeros(self.Ntaus, dtype=np.float16)
            tau_rawIsodR03 = np.zeros(self.Ntaus, dtype=np.float16)
            tau_rawMVAoldDM2017v2 = np.zeros(self.Ntaus, dtype=np.float16)
            tau_rawMVAoldDMdR032017v2 = np.zeros(self.Ntaus, dtype=np.float16)
            tauv = ROOT.TLorentzVector()
            tauIdx = 0
            for tau in taus:
                if tauIdx == self.Ntaus:
                    break
                tauv.SetPtEtaPhiM(tau.pt, tau.eta, tau.phi, tau.mass)
                if jetv.DeltaR(tauv) < 0.8:
                    tau_charge[tauIdx] = tau.charge
                    tau_chargedIso[tauIdx] = tau.chargedIso / tau.pt
                    tau_dxy[tauIdx] = tau.dxy
                    tau_dz[tauIdx] = tau.dz
                    tau_eta[tauIdx] = tau.eta - jeta
                    tau_leadTkDeltaEta[tauIdx] = tau.leadTkDeltaEta
                    tau_leadTkDeltaPhi[tauIdx] = tau.leadTkDeltaPhi
                    tau_leadTkPtOverTauPt[tauIdx] = tau.leadTkPtOverTauPt
                    tau_mass[tauIdx] = tau.mass
                    tau_neutralIso[tauIdx] = tau.neutralIso / tau.pt
                    tau_phi[tauIdx] = signedDeltaPhi(tau.phi, jphi)
                    tau_photonsOutsideSignalCone[tauIdx] = tau.photonsOutsideSignalCone
                    tau_pt[tauIdx] = tau.pt / jpt
                    tau_rawAntiEle[tauIdx] = tau.rawAntiEle
                    tau_rawIso[tauIdx] = tau.rawIso / tau.pt
                    tau_rawIsodR03[tauIdx] = tau.rawIsodR03
                    tau_rawMVAoldDM2017v2[tauIdx] = tau.rawMVAoldDM2017v2
                    tau_rawMVAoldDMdR032017v2[tauIdx] = tau.rawMVAoldDMdR032017v2
                    tauIdx += 1

                    ##find candidates associated to jet
            candrange = range(pf_idx, pf_idx + jet.nPFConstituents)

            ##Fill PF candidates
            pfpt = np.zeros(self.Nparts, dtype=np.float16)
            pfeta = np.zeros(self.Nparts, dtype=np.float16)
            pfphi = np.zeros(self.Nparts, dtype=np.float16)
            pftrk = np.zeros(self.Nparts, dtype=np.float16)
            pfpup = np.zeros(self.Nparts, dtype=np.float16)
            pfpupnolep = np.zeros(self.Nparts, dtype=np.float16)
            pfq = np.zeros(self.Nparts, dtype=np.float16)
            pfid = np.zeros(self.Nparts, dtype=np.float16)
            pfdz = np.zeros(self.Nparts, dtype=np.float16)
            pfdxy = np.zeros(self.Nparts, dtype=np.float16)
            pfdxyerr = np.zeros(self.Nparts, dtype=np.float16)
            arrIdx = 0
            for ip, part in enumerate(pfcands):
                if ip not in candrange: continue
                if arrIdx == self.Nparts: break
                pfpt[arrIdx] = part.pt / jpt
                pfeta[arrIdx] = part.eta - jeta
                pfphi[arrIdx] = signedDeltaPhi(part.phi, jphi)
                pfpup[arrIdx] = part.puppiWeight
                pfpupnolep[arrIdx] = part.puppiWeightNoLep
                pfq[arrIdx] = part.charge
                pfid[arrIdx] = part.pdgId
                pfdz[arrIdx] = part.dz
                pfdxy[arrIdx] = part.d0
                pfdxyerr[arrIdx] = part.d0Err
                pftrk[arrIdx] = part.trkChi2
                arrIdx += 1

            # print(pfpt,pfeta,pfphi,pfdz,pfd0)
            ##define and reshape features
            pfData = np.vstack([pfpt, pfeta, pfphi, pfq, pfdz, pfdxy, pfdxyerr, pfpup, pfpupnolep, pfid])
            pfData = np.transpose(pfData)
            pfData = np.expand_dims(pfData,axis=0)
            svData = np.vstack([svdlen,svdlenSig, svdxy, svdxySig, svchi2, svpAngle, svx, svy, svz, svpt, svmass, sveta, svphi])
            svData = np.transpose(svData)
            svData = np.expand_dims(svData, axis=0)
            #["MET_covXX","MET_covXY","MET_covYY","MET_phi","MET_pt","MET_significance","PuppiMET_pt","PuppiMET_phi","fj_eta","fj_phi","fj_msd","fj_pt"]
            #evtData = np.array([met.covXX,met.covXY,met.covYY,met.phi,met.pt,met.significance,pupmet.pt,pupmet.phi,jeta,jphi,jmsd,jpt])
            evtData = np.array([met.covXX,met.covXY,met.covYY,signedDeltaPhi(met.phi,jphi),met.pt,met.significance,pupmet.pt,signedDeltaPhi(pupmet.phi,jphi),jeta,jphi,jmsd,jpt])
            evtData_sangeon = np.array([met.covXX,met.covXY,met.covYY,signedDeltaPhi(met.phi,jphi),met.pt,met.significance,pupmet.pt,signedDeltaPhi(pupmet.phi,jphi),jmsd,jpt,jeta,jphi])
            evtData = np.expand_dims(evtData,axis=0)
            evtData_sangeon = np.expand_dims(evtData_sangeon,axis=0)

            tauData = np.vstack([tau_charge, tau_chargedIso, tau_dxy, tau_dz, tau_eta, tau_leadTkDeltaEta, tau_leadTkDeltaPhi, tau_leadTkPtOverTauPt, tau_mass, tau_neutralIso, tau_phi, tau_photonsOutsideSignalCone, tau_pt, tau_rawAntiEle, tau_rawIso, tau_rawIsodR03, tau_rawMVAoldDM2017v2, tau_rawMVAoldDMdR032017v2])
            tauData = np.transpose(tauData)
            tauData = np.expand_dims(tauData, axis=0)

            idconv = {211.:1, 13.:2,  22.:3,  11.:4, 130.:5, 1.:6, 2.:7, 3.:8, 4.:9,
                5.:10, -211.:1, -13.:2,
                -11.:4, -1.:-6, -2.:7, -3.:8, -4.:9, -5.:10, 0.:0}
            pfData[:,:,-1] = np.vectorize(idconv.__getitem__)(pfData[:,:,-1])

            Ztagger_pred = self.Ztagger.predict([pfData, svData])
            Ztagger_Zee[0] = float(Ztagger_pred[0][0])
            Ztagger_Zmm[0] = float(Ztagger_pred[0][1])
            Ztagger_Zhh[0] = float(Ztagger_pred[0][2])
            Ztagger_Zhe[0] = float(Ztagger_pred[0][3])
            Ztagger_Zhm[0] = float(Ztagger_pred[0][4])

            MassReg_hadhad_pred = self.massreg_hadhad.predict([pfData, svData, evtData_sangeon])
            MassReg_hadhad_mass[0] = float(MassReg_hadhad_pred[0][0])
            MassReg_hadhad_pt[0] = float(MassReg_hadhad_pred[0][1])

            MassReg_hadel_pred = self.massreg_hadel.predict([pfData, svData, evtData_sangeon])
            MassReg_hadel_mass[0] = float(MassReg_hadel_pred[0][0])
            MassReg_hadel_pt[0] = float(MassReg_hadel_pred[0][1])

            MassReg_hadmu_pred = self.massreg_hadmu.predict([pfData, svData, evtData_sangeon])
            MassReg_hadmu_mass[0] = float(MassReg_hadmu_pred[0][0])
            MassReg_hadmu_pt[0] = float(MassReg_hadmu_pred[0][1])

            idlist = np.abs(pfData[:,:,-1]).astype(int)
            pfData = np.concatenate([pfData[:,:,:-1],np.eye(11)[idlist]],axis=-1)#relies on number of IDs being 11, be careful

            IN_hadhad_v4p1[0] = float(self.model4p1_hadhad.predict([pfData, svData]))
            IN_hadel_v4p1[0] = float(self.model4p1_hadel.predict([pfData, svData]))
            IN_hadmu_v4p1[0] = float(self.model4p1_hadmu.predict([pfData, svData]))

            PostTagger_hadhad_v1p1[0] = float(self.postTagger1p1_hadhad.predict([tauData, IN_hadhad_v4p1]))
            PostTagger_hadel_v1p1[0] = float(self.postTagger1p1_hadel.predict([tauData, IN_hadel_v4p1]))
            PostTagger_hadmu_v1p1[0] = float(self.postTagger1p1_hadmu.predict([tauData, IN_hadmu_v4p1]))

            #self.log_pf.append(pfData)
            #self.log_sv.append(svData)
            #self.log_evt.append(evtData)
            #self.log_mreg.append(np.array([MassReg_hadhad[0], MassReg_hadel[0], MassReg_hadmu[0]]))

            #with open('test.npy', 'wb') as f:
            #    np.save(f, np.vstack(self.log_pf))
            #    np.save(f, np.vstack(self.log_sv))
            #    np.save(f, np.vstack(self.log_evt))
            #    np.save(f, np.vstack(self.log_mreg))
                #np.save(f, pfData)
                #np.save(f, svData)
                #np.save(f, evtData)
                #np.save(f, np.array([MassReg_hadhad[0], MassReg_hadel[0], MassReg_hadmu[0]]))
                #np.save(f, self.massreg_hadhad.get_weights())
                #np.save(f, self.massreg_hadel.get_weights())
                #np.save(f, self.massreg_hadmu.get_weights())

            # assert abs( 1 - float(self.model.predict(X)[0,1]) - float(self.model.predict(X)[0,0])) < 0.02
            # print(X,IN_hadhad_v4p1[0], GRU_hadel_v6p1[0])

        self.out.fillBranch("IN_hadhad_v4p1", IN_hadhad_v4p1)
        self.out.fillBranch("IN_hadel_v4p1", IN_hadel_v4p1)
        self.out.fillBranch("IN_hadmu_v4p1", IN_hadmu_v4p1)

        self.out.fillBranch("Ztagger_Zee", Ztagger_Zee)
        self.out.fillBranch("Ztagger_Zmm", Ztagger_Zmm)
        self.out.fillBranch("Ztagger_Zhh", Ztagger_Zhh)
        self.out.fillBranch("Ztagger_Zhe", Ztagger_Zhe)
        self.out.fillBranch("Ztagger_Zhm", Ztagger_Zhm)

        self.out.fillBranch("PostTagger_hadhad_v1p1", PostTagger_hadhad_v1p1)
        self.out.fillBranch("PostTagger_hadel_v1p1", PostTagger_hadel_v1p1)
        self.out.fillBranch("PostTagger_hadmu_v1p1", PostTagger_hadmu_v1p1)

        self.out.fillBranch("MassReg_hadhad_mass", MassReg_hadhad_mass)
        self.out.fillBranch("MassReg_hadhad_pt", MassReg_hadhad_pt)
        self.out.fillBranch("MassReg_hadel_mass", MassReg_hadel_mass)
        self.out.fillBranch("MassReg_hadel_pt", MassReg_hadel_pt)
        self.out.fillBranch("MassReg_hadmu_mass", MassReg_hadmu_mass)
        self.out.fillBranch("MassReg_hadmu_pt", MassReg_hadmu_pt)
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def signedDeltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    if (dPhi < -np.pi):
        dPhi = 2 * np.pi + dPhi
    elif (dPhi > np.pi):
        dPhi = -2 * np.pi + dPhi
    return dPhi


inferencer = lambda: inferencerClass(jetSelection=lambda j: j.pt >= 0.)  # and j.msoftdrop >= 40.)
