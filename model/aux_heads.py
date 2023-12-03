from torch.nn import MSELoss
import torch.nn as nn
import torch
from model.openfold_multimer.primitives import Linear
from model.openfold.evoformer import EvoformerStack
from typing import Optional

class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins=50, c_in=384, c_hidden=128):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s

class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s=384, c_out=37, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits

class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z=128, no_bins=64, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
    
def compute_fape2(
    pred_frames,
    target_frames,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
    hh_mask=None
) -> torch.Tensor:
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * hh_mask

    normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + torch.sum(hh_mask, dim=(-1, -2)))
    return normed_error
 
def multimer_backbone_loss(
    backbone_affine_tensor: torch.Tensor,
    backbone_affine_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    inter_mask=None,
    intra_mask=None,
    inter_mask_idp=None,
    **kwargs,
) -> torch.Tensor:
    pred_aff = T.from_tensor(traj)
    gt_aff = T.from_tensor(backbone_affine_tensor)

    #inter chain fape
    inter_fape_loss = compute_fape2(
        pred_aff,
        gt_aff[None],
        backbone_affine_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_affine_mask[None],
        l1_clamp_distance=30,
        length_scale=10,
        eps=eps,
        hh_mask=inter_mask
    )

    #intra chain fape
    intra_fape_loss = compute_fape2(
        pred_aff,
        gt_aff[None],
        backbone_affine_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_affine_mask[None],
        l1_clamp_distance=10,
        length_scale=10,
        eps=eps,
        hh_mask=intra_mask
    )

    inter_fape_idp_loss = compute_fape2(
        pred_aff,
        gt_aff[None],
        backbone_affine_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_affine_mask[None],
        l1_clamp_distance=30,
        length_scale=10,
        eps=eps,
        hh_mask=inter_mask_idp
    )

    fape_loss = inter_fape_loss + intra_fape_loss + inter_fape_idp_loss
    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    idp_loss = inter_fape_loss + inter_fape_idp_loss
    return fape_loss, idp_loss

class BasicBlock2D(nn.Module):
    def __init__(self, channels=128, kernel_size=3, padding=1, dropout=0.1, stride=1, dilation=1):

        super(BasicBlock2D, self).__init__()

        padding = padding * dilation
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
        self.elu2 = nn.ELU()
        
    def forward(self, x):
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.elu2(out)

        return out

class ContactBlock2D(nn.Module):
    def __init__(self, channels=128, kernel_size=3, padding=1, dropout=0.1, stride=1, dilation=1, in_channels=1):

        super(ContactBlock2D, self).__init__()

        padding = padding * dilation
        
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size, stride, padding, dilation)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
        self.elu2 = nn.ELU()
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.elu2(out)

        return out

class ContactHead(nn.Module):
    def __init__(self, num_blocks=5):

        super(ContactHead, self).__init__()

        self.residual_blocks = nn.ModuleList([])
        for i in range(num_blocks):
           self.residual_blocks.append(BasicBlock2D())
        self.pred = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pred2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pred3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1)


    def mask2d(self, emb, mask):
        emb = emb.squeeze(0)
        mask_b = emb[mask.bool()]
        mask_b = mask_b.permute(1, 0)
        mask_b = mask_b[mask.bool()]
        mask_b = mask_b.permute(1, 0)

        return mask_b.unsqueeze(0).contiguous() 

    def mask3d(self, emb, mask):
        emb = emb.squeeze(0)
        mask_b = emb[mask.bool()]
        mask_b = mask_b.permute(1, 0, 2)
        mask_b = mask_b[mask.bool()]
        mask_b = mask_b.permute(1, 0, 2)

        return mask_b.unsqueeze(0).contiguous() 

    def forward2(self, input, target, alpha=0.5, smooth=1, seq_mask=None, missing=None):    
        input = self.mask2d(input, missing)
        target = self.mask2d(target, seq_mask)

        #print(input.size())
        input_pos = input.view(-1)
        target_pos = target.view(-1)
        true_pos = torch.sum(input_pos * target_pos)
        false_neg = torch.sum(target_pos * (1 - input_pos))
        false_pos = torch.sum((1 - target_pos) * input_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    def tversky_coeff(self, input, target, seq_mask=None, missing=None):
        """tversky coeff for batches"""
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            s = s + self.forward2(c[0], c[1], alpha=0.7, seq_mask=seq_mask[i], missing=missing[i])

        return s / (i + 1)

    def forward(self, embedding, contacts, seq_mask=None, missing=None, emb=None, inter_mask=None, h_mask=None):
        
        embedding = embedding.permute(0, 3, 1, 2)
        for layer in self.residual_blocks:
            embedding = layer(embedding)
        logits = self.pred(embedding)
        logits = logits + logits.transpose(-1, -2)
        pred = torch.sigmoid(logits)

        logits_contacts = self.pred2(embedding)
        logits_contacts = logits_contacts + logits_contacts.transpose(-1, -2)
        pred2 = torch.sigmoid(logits_contacts)

        logits_h = self.pred3(embedding)
        logits_h = logits_h + logits_h.transpose(-1, -2)
        pred3 = torch.sigmoid(logits_h)
        
        contact_loss = None
        if not contacts is None:
            contact_loss = 1 - self.tversky_coeff(pred2, contacts, seq_mask=seq_mask, missing=missing)
            contact_loss = contact_loss[0]

            inter_contact_loss = 1 - self.tversky_coeff(pred, contacts * inter_mask, seq_mask=seq_mask, missing=missing)
            inter_contact_loss = inter_contact_loss[0]

            h_contact_loss = 1 - self.tversky_coeff(pred3, h_mask, seq_mask=seq_mask, missing=missing)
            h_contact_loss = h_contact_loss[0]
            contact_loss = 0.7 * inter_contact_loss + 0.2 * contact_loss + 0.1 * h_contact_loss
        threshhold = 0.5
        ones, zeros = torch.ones(logits.size()).to(embedding.device), torch.zeros(logits.size()).to(embedding.device)
        contacts_map = torch.where(pred > threshhold, ones, zeros)

        if contacts is None:
            return embedding.permute(0, 2, 3, 1), 0, contacts_map
        embedding = embedding.permute(0, 2, 3, 1)
        return embedding, contact_loss, contacts_map

def embedding_loss(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    seq_mask: torch.Tensor = None,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    pair_mask = None,
    **kwargs,
) -> torch.Tensor:
    mask = pair_mask
    error_dist = torch.sqrt(
        torch.mean((pred_emb - target_emb) ** 2, dim=-1) + eps
    )
    
    normed_error = error_dist * mask
    normed_error = torch.sum(normed_error, dim=(-1, -2, -3)) / (eps + torch.sum(mask, dim=(-1, -2, -3)))
    return normed_error

def embedding_loss_single(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    seq_mask: torch.Tensor = None,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    pair_mask = None,
    **kwargs,
) -> torch.Tensor:
    mask = seq_mask
    error_dist = torch.sqrt(
        torch.mean((pred_emb - target_emb) ** 2, dim=-1) + eps
    )
    
    normed_error = error_dist * mask
    normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + torch.sum(mask, dim=(-1, -2)))
    return normed_error

def center_of_mass_loss(
    pred_ca_positions: torch.Tensor,  # (b, N, 3) rearranged
    gt_ca_positions: torch.Tensor,  # (b, N, 3) rearranged
    eps=1e-9,
) -> torch.Tensor:
    #N*N
    num_chains =  gt_ca_positions.size(1)
    dist_pred = torch.squeeze(torch.cdist(pred_ca_positions, pred_ca_positions))
    dist_gt = torch.squeeze(torch.cdist(gt_ca_positions, gt_ca_positions))
    center_loss = (dist_pred - dist_gt + 4) / 20

    center_loss = torch.where(center_loss < 0, center_loss**2, torch.zeros(center_loss.size()).to(device=center_loss.device))
    #print(center_loss)
    return torch.sum(center_loss) / (num_chains * (num_chains - 1) + eps)

class KDHead(nn.Module):
    def __init__(self, args, num_blocks=5):
        super(KDHead, self).__init__()
        self.residual_blocks = nn.ModuleList([])
        self.contact_res = ContactBlock2D()
        for _ in range(num_blocks):
           self.residual_blocks.append(BasicBlock2D(dropout=0.1))

        self.fc = nn.Sequential(
            Linear(128, 128, init="relu"),
            nn.ReLU(inplace=True),
            Linear(128, 128, init="relu"),
            nn.ReLU(inplace=True),
            Linear(128, 128, init="final")
        )
        self.pae_block = ContactBlock2D()

        #evoformer
        self.evoformer = EvoformerStack(no_blocks=num_blocks)

    def mse_loss(self, pred, gt):
        loss = MSELoss()
        return loss(pred, gt)

    def forward(self, embedding, single=None, batch_gt=None):
        contact_emb = None
        chunk_size = None
        if not self.training:
            chunk_size = 8
        single = single.unsqueeze(1)
        single, embedding = self.evoformer(
                single,
                embedding,
                msa_mask=None,
                pair_mask=None,
                chunk_size=chunk_size,
                _mask_trans=False,
        )
        single = single.squeeze(1)
        return single, embedding, contact_emb

class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z=128, no_bins=64, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        return logits
    