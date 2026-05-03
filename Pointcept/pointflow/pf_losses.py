from __future__ import annotations
import torch


def apply_se3_to_points(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    """Apply SE(3) to points.

    Args:
      points: [...,3]
      R:      [...,3,3]
      t:      [...,3]

    Returns:
      transformed points with same leading dims as input points.
    """
    return torch.einsum("...ij,...nj->...ni", R, points) + t.unsqueeze(-2)


def inv_se3(R: torch.Tensor, t: torch.Tensor):
    """Invert SE(3) given rotation R and translation t."""
    Rinv = R.transpose(-1, -2)
    tinv = -torch.einsum("...ij,...j->...i", Rinv, t)
    return Rinv, tinv


def fk_warp_consistency_loss(
    xyz_pred: torch.Tensor,
    segment: torch.Tensor,
    body_xpos: torch.Tensor,
    body_xmat: torch.Tensor,
    robust_delta: float = 0.01,
):
    """Segmented rigid-body warp consistency loss.

    Encourage points on each body to move according to body FK pose:
      T_b(t-1)^{-1} T_b(t) P_b(t)  ~  P_b(t-1)

    Args:
      xyz_pred:  [B,T,N,3] predicted world points
      segment:   [B,N] or [N] int body label in [0,nb-1]
      body_xpos: [B,T,nb,3] world positions
      body_xmat: [B,T,nb,9] world rotations (flattened row-major)

    Returns:
      scalar loss
    """
    if xyz_pred.ndim != 4 or xyz_pred.shape[-1] != 3:
        raise ValueError(f"xyz_pred must be [B,T,N,3], got {tuple(xyz_pred.shape)}")

    B, T, N, _ = xyz_pred.shape
    nb = body_xpos.shape[2]

    if segment.ndim == 1:
        seg = segment.view(1, -1).expand(B, -1)
    elif segment.ndim == 2:
        seg = segment
    else:
        raise ValueError(f"segment must be [N] or [B,N], got {tuple(segment.shape)}")

    if seg.shape[1] != N:
        raise ValueError(f"segment N mismatch: seg={seg.shape[1]} vs xyz N={N}")

    R = body_xmat.view(B, T, nb, 3, 3)
    t_w = body_xpos

    total = xyz_pred.new_tensor(0.0)
    denom = 0

    for bi in range(nb):
        m = seg == int(bi)  # [B,N]
        if not m.any():
            continue

        for b in range(B):
            idx = torch.nonzero(m[b], as_tuple=False).squeeze(1)
            if idx.numel() < 8:
                continue

            P = xyz_pred[b, :, idx, :]  # [T,Ni,3]
            Rb = R[b, :, bi, :, :]  # [T,3,3]
            tb = t_w[b, :, bi, :]  # [T,3]

            # inv(T(t))
            Rinv_t, tinv_t = inv_se3(Rb.unsqueeze(0), tb.unsqueeze(0))
            Rinv_t = Rinv_t.squeeze(0)
            tinv_t = tinv_t.squeeze(0)

            # P_body(t) = inv(T(t)) * P_w(t)
            P_body = torch.einsum("tij,tnj->tni", Rinv_t, P) + tinv_t.unsqueeze(1)

            # P_warp(t->t-1) = T(t-1) * P_body(t)
            R_prev = Rb[:-1]
            t_prev = tb[:-1]
            P_body_next = P_body[1:]
            P_warp = torch.einsum("tij,tnj->tni", R_prev, P_body_next) + t_prev.unsqueeze(1)

            P_tgt = P[:-1]
            diff = P_warp - P_tgt  # [T-1,Ni,3]

            abs_diff = diff.abs()
            quad = torch.minimum(abs_diff, diff.new_tensor(float(robust_delta)))
            lin = abs_diff - quad
            huber = 0.5 * quad * quad / float(robust_delta) + lin

            total = total + huber.mean()
            denom += 1

    if denom == 0:
        return total
    return total / float(denom)


def rigid_pairwise_distance_consistency_loss(
    xyz_pred: torch.Tensor,
    segment: torch.Tensor,
    num_pairs_per_body: int = 1024,
    robust_delta: float = 0.01,
    min_points_per_body: int = 8,
    generator: torch.Generator | None = None,
):
    """Rigid consistency loss without using body pose.

    For each body, randomly sample point pairs (i,j) within that body and enforce
    pairwise distances to be constant across adjacent frames:

      ||p_i(t) - p_j(t)||  ~  ||p_i(t-1) - p_j(t-1)||

    This matches your point extraction pipeline where each point index corresponds
    to a fixed sampled surface point on a rigid link (body).

    Args:
      xyz_pred: [B,T,N,3]
      segment:  [B,N] or [N] int body label
      num_pairs_per_body: number of pairs sampled per body (per batch element)
      robust_delta: huber delta in meters
      min_points_per_body: skip bodies with fewer points
      generator: optional torch random generator for determinism

    Returns:
      scalar loss
    """
    if xyz_pred.ndim != 4 or xyz_pred.shape[-1] != 3:
        raise ValueError(f"xyz_pred must be [B,T,N,3], got {tuple(xyz_pred.shape)}")

    B, T, N, _ = xyz_pred.shape

    if segment.ndim == 1:
        seg = segment.view(1, -1).expand(B, -1)
    elif segment.ndim == 2:
        seg = segment
    else:
        raise ValueError(f"segment must be [N] or [B,N], got {tuple(segment.shape)}")

    if seg.shape[1] != N:
        raise ValueError(f"segment N mismatch: seg={seg.shape[1]} vs xyz N={N}")

    # unique bodies present (assume contiguous small ids but not required)
    body_ids = torch.unique(seg).detach().to(torch.long)

    total = xyz_pred.new_tensor(0.0)
    denom = 0

    for bi_t in body_ids.tolist():
        bi = int(bi_t)
        if bi < 0:
            continue

        m = seg == bi  # [B,N]
        if not m.any():
            continue

        for b in range(B):
            idx = torch.nonzero(m[b], as_tuple=False).squeeze(1)
            ni = int(idx.numel())
            if ni < int(min_points_per_body):
                continue

            # sample pairs (with replacement)
            # i,j in [0,ni)
            if int(num_pairs_per_body) <= 0:
                continue

            ii = torch.randint(0, ni, (int(num_pairs_per_body),), device=xyz_pred.device, generator=generator)
            jj = torch.randint(0, ni, (int(num_pairs_per_body),), device=xyz_pred.device, generator=generator)

            pi = xyz_pred[b, :, idx[ii], :]  # [T,P,3]
            pj = xyz_pred[b, :, idx[jj], :]  # [T,P,3]

            # distances per frame
            d = torch.linalg.norm(pi - pj, dim=-1)  # [T,P]
            dd = d[1:] - d[:-1]  # [T-1,P]

            abs_diff = dd.abs()
            quad = torch.minimum(abs_diff, abs_diff.new_tensor(float(robust_delta)))
            lin = abs_diff - quad
            huber = 0.5 * quad * quad / float(robust_delta) + lin

            total = total + huber.mean()
            denom += 1

    if denom == 0:
        return total
    return total / float(denom)
