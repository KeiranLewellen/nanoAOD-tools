
import ROOT
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True
from array import array
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
#from keras.models import load_model
import json
import os
#import tensorflow as tf
import onnxruntime as rt

class inferencerClass(Module):
    def __init__(self, jetSelection):
        self.jetSel = jetSelection
        self.Nparts = 30
        self.Nsvs = 5
        self.Ntaus = 3
        base = os.environ['CMSSW_BASE']

        self.options = rt.SessionOptions() 
        self.options.intra_op_num_threads = 1
        self.options.inter_op_num_threads = 1

        self.model5p1_hadhad_multi_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/IN_hadhad_v5p1_multiclass,on_QCD_WJets_noLep,fillFactor=1:2:1,eventData,take_1,model.onnx', self.options)

        self.IN_hadel_v5p1_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/IN_hadel_v5p1,on_TTbar_WJets,ohe,eventData,take_1,model.onnx', self.options)
        self.IN_hadmu_v5p1_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/IN_hadmu_v5p1,on_TTbar_WJets,ohe,eventData,take_1,model.onnx', self.options)

        self.Ztagger_Zee_Zhe_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/IN_Zhe_v5p1,on_Zee_oneEl_Zhe,ohe,eventZ,take_2,model.onnx', self.options)
        self.Ztagger_Zmm_Zhm_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/IN_Zhm_v5p1,on_Zmm_oneMu_Zhm,ohe,eventZ,take_1,model.onnx', self.options)

        self.MassReg_hadhad_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/hadhad_H20000_Z25000_Lambda0.01_FLAT500k_genPtCut400.onnx', self.options)
        self.MassReg_hadel_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/hadel_H15000_Z15000_Lambda0.1_hadel_FLAT300k_genPtCut300.onnx', self.options)
        self.MassReg_hadmu_session = rt.InferenceSession(base+'/src/PhysicsTools/NanoAODTools/data/hadmu_H9000_Z15000_Lambda0.01_hadmu_FLAT300k_genPtCut300.onnx', self.options)

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


        #self.out.branch("IN_hadhad_v4p1", "F", 1)
        self.out.branch("IN_hadel_v5p1", "F", 1)
        self.out.branch("IN_hadmu_v5p1", "F", 1)

        self.out.branch("Ztagger_v5p1_Zee_Zhe", "F", 1)
        self.out.branch("Ztagger_v5p1_Zmm_Zhm", "F", 1)

        self.out.branch("IN_hadhad_v5p1_multi_Higgs", "F", 1)
        self.out.branch("IN_hadhad_v5p1_multi_QCD", "F", 1)
        self.out.branch("IN_hadhad_v5p1_multi_WJets", "F", 1)

        self.out.branch("MassReg_hadhad_mass", "F", 1)
        self.out.branch("MassReg_hadel_mass", "F", 1)
        self.out.branch("MassReg_hadmu_mass", "F", 1)

        self.out.branch("MassReg_hadhad_pt", "F", 1)
        self.out.branch("MassReg_hadel_pt", "F", 1)
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

        Ztagger_Zee_Zhe = np.full(1, -1., dtype=np.float32)
        Ztagger_Zmm_Zhm = np.full(1, -1., dtype=np.float32)

        IN_hadel_v5p1 = np.full(1, -1., dtype=np.float32)
        IN_hadmu_v5p1 = np.full(1, -1., dtype=np.float32)

        IN_hadhad_v5p1_multi_Higgs = np.full(1, -1., dtype=np.float32)
        IN_hadhad_v5p1_multi_QCD = np.full(1, -1., dtype=np.float32)
        IN_hadhad_v5p1_multi_WJets = np.full(1, -1., dtype=np.float32)

        MassReg_hadhad_mass = np.full(1, -1., dtype=np.float32)
        MassReg_hadel_mass = np.full(1, -1., dtype=np.float32)
        MassReg_hadmu_mass = np.full(1, -1., dtype=np.float32)

        MassReg_hadhad_pt   = np.full(1, -1., dtype=np.float32)
        MassReg_hadel_pt = np.full(1, -1., dtype=np.float32)
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


            if jmsd == 0:
                jLSmsd = np.inf
            else:
                jLSmsd = jLSmsd/jmsd

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
            pfvtx = np.zeros(self.Nparts, dtype=np.float16)
            arrIdx = 0
            jMuonEnergy = np.float16(0)
            jElectronEnergy = np.float16(0)
            jPhotonEnergy = np.float16(0)
            jChargedHadronEnergy = np.float16(0)
            jNeutralHadronEnergy = np.float16(0)
            jMuonNum = np.float16(0)
            jElectronNum = np.float16(0)
            jPhotonNum = np.float16(0)
            jChargedHadronNum = np.float16(0)
            jNeutralHadronNum = np.float16(0)
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
                pfvtx[arrIdx] = part.vtxChi2
                if part.pdgId in [-13., 13.]:
                    jMuonEnergy += part.pt
                    jMuonNum += 1
                if part.pdgId in [-11., 11.]:
                    jElectronEnergy += part.pt
                    jElectronNum += 1
                if part.pdgId in [22.]:
                    jPhotonEnergy += part.pt
                    jPhotonNum += 1
                if part.pdgId in [-211., 211.]:
                    jChargedHadronEnergy += part.pt
                    jChargedHadronNum += 1
                if part.pdgId in [-111., 111.,  130.]:
                    jNeutralHadronEnergy += part.pt
                    jNeutralHadronNum += 1
                arrIdx += 1


            # Define and reshape features

            pfData = np.vstack([pfpt, pfeta, pfphi, pfq, pfdz, pfdxy, pfdxyerr, pfpup, pfpupnolep, pfid])
            pfData = np.transpose(pfData)
            pfData = np.expand_dims(pfData,axis=0)

            pfDataMore = np.vstack([pfpt, pfeta, pfphi, pfq, pfdz, pfdxy, pfdxyerr, pfpup, pfpupnolep, pfid, pftrk, pfvtx])
            pfDataMore = np.transpose(pfDataMore)
            pfDataMore = np.expand_dims(pfDataMore, axis=0)

            svData = np.vstack([svdlen,svdlenSig, svdxy, svdxySig, svchi2, svpAngle, svx, svy, svz, svpt, svmass, sveta, svphi])
            svData = np.transpose(svData)
            svData = np.expand_dims(svData, axis=0)

            #["MET_covXX","MET_covXY","MET_covYY","MET_phi","MET_pt","MET_significance","PuppiMET_pt","PuppiMET_phi","fj_eta","fj_phi","fj_msd","fj_pt"]
            #evtData = np.array([met.covXX,met.covXY,met.covYY,met.phi,met.pt,met.significance,pupmet.pt,pupmet.phi,jeta,jphi,jmsd,jpt])
            #evtData = np.array([met.covXX,met.covXY,met.covYY,signedDeltaPhi(met.phi,jphi),met.pt,met.significance,pupmet.pt,signedDeltaPhi(pupmet.phi,jphi),jeta,jphi,jmsd,jpt])

            evtData_reg = np.array([met.covXX,met.covXY,met.covYY,signedDeltaPhi(met.phi,jphi),met.pt,met.significance,pupmet.pt,signedDeltaPhi(pupmet.phi,jphi),jmsd,jpt,jeta,jphi])
            evtData_reg = np.expand_dims(evtData_reg,axis=0)

            evtZ = np.array([jMuonEnergy/jpt, jElectronEnergy/jpt, jPhotonEnergy/jpt, jChargedHadronEnergy/jpt, jNeutralHadronEnergy/jpt, jMuonNum, jElectronNum, jPhotonNum, jChargedHadronNum, jNeutralHadronNum, jLSpt/jpt])
            evtZ = np.expand_dims(evtZ,axis=0)

            evtData = np.array(
                [jMuonEnergy / jpt, jElectronEnergy / jpt, jPhotonEnergy / jpt, jChargedHadronEnergy / jpt,
                 jNeutralHadronEnergy / jpt, jMuonNum, jElectronNum, jPhotonNum, jChargedHadronNum, jNeutralHadronNum,
                 jLSpt / jpt, jLSmsd])
            evtData = np.expand_dims(evtData, axis=0)


            tauData = np.vstack([tau_charge, tau_chargedIso, tau_dxy, tau_dz, tau_eta, tau_leadTkDeltaEta, tau_leadTkDeltaPhi, tau_leadTkPtOverTauPt, tau_mass, tau_neutralIso, tau_phi, tau_photonsOutsideSignalCone, tau_pt, tau_rawAntiEle, tau_rawIso, tau_rawIsodR03, tau_rawMVAoldDM2017v2, tau_rawMVAoldDMdR032017v2])
            tauData = np.transpose(tauData)
            tauData = np.expand_dims(tauData, axis=0)

            idconv = {211.:1, 13.:2,  22.:3,  11.:4, 130.:5, 1.:6, 2.:7, 3.:8, 4.:9,
                5.:10, -211.:1, -13.:2,
                -11.:4, -1.:-6, -2.:7, -3.:8, -4.:9, -5.:10, 0.:0}
            pfData[:,:,-1] = np.vectorize(idconv.__getitem__)(pfData[:,:,-1])
            pfDataMore[:, :, -3] = np.vectorize(idconv.__getitem__)(pfDataMore[:, :, -3]) # remember which column the pfids are in

            pfData_reg = pfData

            idlist = np.abs(pfData[:,:,-1]).astype(int)
            pfData = np.concatenate([pfData[:,:,:-1],np.eye(11)[idlist]],axis=-1)  # relies on number of IDs being 11, be careful

            idlistMore = np.abs(pfDataMore[:,:,-3]).astype(int)
            pfDataMore = np.concatenate([pfDataMore[:, :, :-3], np.eye(11)[idlistMore],pfDataMore[:, :, -2:]],axis=-1)  # relies on number of IDs being 11, be careful

            pfData = pfData.astype(np.float32)
            pfDataMore = pfDataMore.astype(np.float32)
            svData = svData.astype(np.float32)
            evtData = evtData.astype(np.float32)
            evtZ = evtZ.astype(np.float32)
            evtData_reg = evtData_reg.astype(np.float32)
            pfData_reg = pfData_reg.astype(np.float32)

            # Performs inference using models

            IN_hadel_v5p1[0] = float(
                self.IN_hadel_v5p1_session.run([self.IN_hadel_v5p1_session.get_outputs()[0].name],
                                                 {self.IN_hadel_v5p1_session.get_inputs()[0].name: evtData,
                                                  self.IN_hadel_v5p1_session.get_inputs()[1].name: pfDataMore,
                                                  self.IN_hadel_v5p1_session.get_inputs()[2].name: svData})[0][0])

            IN_hadmu_v5p1[0] = float(
                self.IN_hadmu_v5p1_session.run([self.IN_hadmu_v5p1_session.get_outputs()[0].name],
                                               {self.IN_hadmu_v5p1_session.get_inputs()[0].name: evtData,
                                                self.IN_hadmu_v5p1_session.get_inputs()[1].name: pfDataMore,
                                                self.IN_hadmu_v5p1_session.get_inputs()[2].name: svData})[0][0])

            Ztagger_Zee_Zhe[0] = float(
                self.Ztagger_Zee_Zhe_session.run([self.Ztagger_Zee_Zhe_session.get_outputs()[0].name],
                                                {self.Ztagger_Zee_Zhe_session.get_inputs()[0].name: evtZ,
                                                 self.Ztagger_Zee_Zhe_session.get_inputs()[1].name: pfDataMore,
                                                 self.Ztagger_Zee_Zhe_session.get_inputs()[2].name: svData})[0][0])
            Ztagger_Zmm_Zhm[0] = float(
                self.Ztagger_Zmm_Zhm_session.run([self.Ztagger_Zmm_Zhm_session.get_outputs()[0].name],
                                                {self.Ztagger_Zmm_Zhm_session.get_inputs()[0].name: evtZ,
                                                 self.Ztagger_Zmm_Zhm_session.get_inputs()[1].name: pfDataMore,
                                                 self.Ztagger_Zmm_Zhm_session.get_inputs()[2].name: svData})[0][0])


            IN_hadhad_v5p1_multi_pred = self.model5p1_hadhad_multi_session.run(
                [self.model5p1_hadhad_multi_session.get_outputs()[0].name],
                {self.model5p1_hadhad_multi_session.get_inputs()[0].name: evtData,
                 self.model5p1_hadhad_multi_session.get_inputs()[1].name: pfDataMore,
                 self.model5p1_hadhad_multi_session.get_inputs()[2].name: svData})

            IN_hadhad_v5p1_multi_Higgs[0] = float(IN_hadhad_v5p1_multi_pred[0][0][0])
            IN_hadhad_v5p1_multi_QCD[0] = float(IN_hadhad_v5p1_multi_pred[0][0][1])
            IN_hadhad_v5p1_multi_WJets[0] = float(IN_hadhad_v5p1_multi_pred[0][0][2])

            MassReg_hadhad_pred = self.MassReg_hadhad_session.run(
                [self.MassReg_hadhad_session.get_outputs()[0].name],
                {self.MassReg_hadhad_session.get_inputs()[0].name: evtData_reg,
                 self.MassReg_hadhad_session.get_inputs()[1].name: pfData_reg,
                 self.MassReg_hadhad_session.get_inputs()[2].name: svData})

            MassReg_hadhad_mass[0] = float(MassReg_hadhad_pred[0][0][0])
            MassReg_hadhad_pt[0] = float(MassReg_hadhad_pred[0][0][1])

            MassReg_hadel_pred = self.MassReg_hadel_session.run(
                [self.MassReg_hadel_session.get_outputs()[0].name],
                {self.MassReg_hadel_session.get_inputs()[0].name: evtData_reg,
                 self.MassReg_hadel_session.get_inputs()[1].name: pfData_reg,
                 self.MassReg_hadel_session.get_inputs()[2].name: svData})

            MassReg_hadel_mass[0] = float(MassReg_hadel_pred[0][0][0])
            MassReg_hadel_pt[0] = float(MassReg_hadel_pred[0][0][1])

            MassReg_hadmu_pred = self.MassReg_hadmu_session.run(
                [self.MassReg_hadmu_session.get_outputs()[0].name],
                {self.MassReg_hadmu_session.get_inputs()[0].name: evtData_reg,
                 self.MassReg_hadmu_session.get_inputs()[1].name: pfData_reg,
                 self.MassReg_hadmu_session.get_inputs()[2].name: svData})

            MassReg_hadmu_mass[0] = float(MassReg_hadmu_pred[0][0][0])
            MassReg_hadmu_pt[0] = float(MassReg_hadmu_pred[0][0][1])

        # Fills out branches

        self.out.fillBranch("IN_hadel_v5p1", IN_hadel_v5p1)
        self.out.fillBranch("IN_hadmu_v5p1", IN_hadmu_v5p1)

        self.out.fillBranch("Ztagger_v5p1_Zee_Zhe", Ztagger_Zee_Zhe)
        self.out.fillBranch("Ztagger_v5p1_Zmm_Zhm", Ztagger_Zmm_Zhm)

        self.out.fillBranch("IN_hadhad_v5p1_multi_Higgs", IN_hadhad_v5p1_multi_Higgs)
        self.out.fillBranch("IN_hadhad_v5p1_multi_QCD", IN_hadhad_v5p1_multi_QCD)
        self.out.fillBranch("IN_hadhad_v5p1_multi_WJets", IN_hadhad_v5p1_multi_WJets)

        self.out.fillBranch("MassReg_hadhad_mass", MassReg_hadhad_mass)
        self.out.fillBranch("MassReg_hadel_mass", MassReg_hadel_mass)
        self.out.fillBranch("MassReg_hadmu_mass", MassReg_hadmu_mass)

        self.out.fillBranch("MassReg_hadhad_pt", MassReg_hadhad_pt)
        self.out.fillBranch("MassReg_hadel_pt", MassReg_hadel_pt)
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
