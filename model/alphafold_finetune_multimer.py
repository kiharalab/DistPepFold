from model.openfold_multimer.structure_module import StructureModule
from model.openfold.loss import (sidechain_loss, 
                                compute_renamed_ground_truth, 
                                find_structural_violations, 
                                violation_loss,
                                supervised_chi_loss,
                                lddt_loss,
                                backbone_loss,
                                compute_plddt,
                                compute_tm)
from model.openfold.feats import atom14_to_atom37
from model.aux_heads import (PerResidueLDDTCaPredictor,
                            ExperimentallyResolvedHead,
                            TMScoreHead,
                            ContactHead,
                            KDHead,
                            ContactBlock2D,
                            BasicBlock2D,
                            multimer_backbone_loss)
import torch
import torch.nn as nn

class AlphaFold_Multimer(nn.Module):
    def __init__(self, args):
        super(AlphaFold_Multimer, self).__init__()
        self.structure_module = StructureModule(trans_scale_factor=args.point_scale, no_blocks=args.ipa_depth, no_heads_ipa=12, c_ipa=16, dropout_rate=0.1) #no_heads_ipa=24, c_ipa=64
        self.plddt =  PerResidueLDDTCaPredictor()
        self.experimentally_resolved = ExperimentallyResolvedHead()
        self.tm = TMScoreHead()
       
        self.args = args

        tmp_block = 5
        self.contact_res = ContactBlock2D()
        self.residual_blocks = nn.ModuleList([])
        for i in range(tmp_block):
           self.residual_blocks.append(BasicBlock2D())

        #including contacts as loss
        if args.contact:
            self.contact = ContactHead(num_blocks=5)
       
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        self.kd_blocks = KDHead(args, num_blocks=5)

    def forward_training(self, embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution, representation=None, emb2=None):
    
        output_bb, outputs = self.structure_module(single_repr, embedding, aatype=aatype, mask=batch_gt_frames['seq_mask'])
       
        pred_frames = torch.stack(output_bb)

        bb_loss = backbone_loss(
            backbone_affine_tensor=batch_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
            backbone_affine_mask=batch_gt_frames['rigidgroups_gt_exists'][..., 0],
            traj=pred_frames,
        )

        multimer_bb_loss, idp_loss = multimer_backbone_loss(
            backbone_affine_tensor=batch_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
            backbone_affine_mask=batch_gt_frames['rigidgroups_gt_exists'][..., 0],
            traj=pred_frames,
            inter_mask=batch_gt['inter_mask'],
            intra_mask=batch_gt['intra_mask'],
            inter_mask_idp=batch_gt['inter_mask_idp']
        )

        rename =compute_renamed_ground_truth(batch_gt, outputs['positions'][-1])
       
        sc_loss = sidechain_loss(
            sidechain_frames=outputs['sidechain_frames'],
            sidechain_atom_pos=outputs['positions'],
            rigidgroups_gt_frames=batch_gt_frames['rigidgroups_gt_frames'],
            rigidgroups_alt_gt_frames=batch_gt_frames['rigidgroups_alt_gt_frames'],
            rigidgroups_gt_exists=batch_gt_frames['rigidgroups_gt_exists'],
            renamed_atom14_gt_positions=rename['renamed_atom14_gt_positions'],
            renamed_atom14_gt_exists=rename['renamed_atom14_gt_exists'],
            alt_naming_is_better=rename['alt_naming_is_better'],
        )
        
        angle_loss = supervised_chi_loss(outputs['angles'],
                                        outputs['unnormalized_angles'],
                                        aatype=aatype,
                                        seq_mask=batch_gt_frames['seq_mask'],
                                        chi_mask=batch_gt_frames['chi_mask'],
                                        chi_angles_sin_cos=batch_gt_frames['chi_angles_sin_cos'],
                                        chi_weight=0.5,
                                        angle_norm_weight=0.01
                                        )
        
        fape = 0.5 * multimer_bb_loss + 1 * sc_loss

        vio_loss = 0
        plddt_loss = 0
        batch_gt.update({'aatype': aatype})
        violation = find_structural_violations(batch_gt, outputs['positions'][-1],
                                            violation_tolerance_factor=12,
                                            clash_overlap_tolerance=1.5)
        violation_loss_ = violation_loss(violation, batch_gt['atom14_atom_exists'])
        vio_loss = torch.mean(violation_loss_)
        #print(violation_loss_)
        fape += 1 * violation_loss_

        lddt = self.plddt(outputs['single'])
        final_position = atom14_to_atom37(outputs['positions'][-1], batch_gt) 
        plddt_loss = lddt_loss(lddt, final_position, 
                                all_atom_positions=batch_gt['all_atom_positions'], 
                                all_atom_mask=batch_gt['all_atom_mask'],
                                resolution=resolution)
        fape += 0.01 * plddt_loss
        fape = torch.mean(fape)
        fape += 0.5 * angle_loss
        
        #scale the loss
        seq_len = torch.mean(batch_gt["seq_length"].float())
        crop_len = torch.tensor(aatype.shape[-1]).to(device=aatype.device)
       
        final_pos = atom14_to_atom37(outputs['positions'][-1], batch_gt)
        final_atom_mask = batch_gt["atom37_atom_exists"]

        ca_gt = batch_gt["all_atom_positions"][..., 1, :]
        ca_pred = final_pos[..., 1, :]

        chain_id = batch_gt['chain_id']
        
        batch_num = chain_id.size(0)
        center_mass_batch = 0
        for select in range(batch_num):
            ca_gt_s = ca_gt[select]
            ca_pred_s = ca_pred[select]
            chain_id_s = chain_id[select]
            num_chains = torch.max(chain_id_s) - torch.min(chain_id_s) + 1
            chain_center_pred,  chain_center_gt = list(), list()
            unique_id = torch.unique(chain_id_s)
           
            for i in range(len(unique_id)):
                chain_idx = unique_id[i]
                idx = (chain_id_s == chain_idx).nonzero().squeeze(-1)
                start, end = idx[0], idx[-1] + 1
                #print(start, end)
                a_pred, a_gt = torch.mean(ca_pred_s[start : end, :], dim=0), torch.mean(ca_gt_s[start : end, :], dim=0)
                #print(a_pred.size())
                chain_center_pred.append(a_pred.unsqueeze(0))
                chain_center_gt.append(a_gt.unsqueeze(0))
            
            pred = torch.cat(chain_center_pred, dim = 0).unsqueeze(0) #center of mass implementation needs batch dimension
            gt = torch.cat(chain_center_gt, dim = 0).unsqueeze(0)
            center_mass = center_of_mass_loss(pred, gt)
            center_mass_batch += center_mass
      
        center_mass = center_mass_batch / batch_num
       
        fape = (fape + center_mass + contact_loss) * torch.sqrt(min(seq_len, crop_len))
        return None, fape, outputs['positions'], vio_loss, angle_loss, plddt_loss, dist_out, \
                torch.mean(bb_loss), torch.mean(sc_loss), hh_loss, hydro_loss, ionic_loss

    def forward_testing(self, pair, single, aatype, protein_test):
        single_repr, embedding, contact_emb = self.kd_blocks(pair, single=single, batch_gt=protein_test)
        output_bb, outputs = self.structure_module(single_repr, embedding, aatype=aatype, mask=None)
        
        target = protein_test['target']
            
        lddt = self.plddt(outputs['single'])
        plddt = compute_plddt(lddt)

        docking_score = 0 # currently the model deos not support docking score
        pae_logits = self.tm(embedding)
        #compute docking socre
        ptm = compute_tm(pae_logits)
        iptm = compute_tm(pae_logits, asym_id=protein_test['chain_idx'], interface=True)
        docking_score = ptm * 0.2 + iptm * 0.8
        print(docking_score)
        return outputs['positions'], plddt, docking_score

    def forward(self, embedding, single_repr, aatype, batch_gt, batch_gt_frames, 
                resolution, representation=None, training=True, protein_test=None):
        if training:
            return self.forward_training(embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution, representation=representation)
        else:
            return self.forward_testing(embedding, single_repr, aatype, protein_test=protein_test)
