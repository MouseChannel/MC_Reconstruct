import math

import torch

specular_epsilon = 1e-4


def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim=-1)


def _dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def bsdf_lambda_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res


def bsdf_ndf_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * math.pi)


def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)


def bsdf_fresnel_shlick(f0, f90, cosTheta):
    # _cosTheta = torch.clamp(cosTheta, min=specular_epsilon, max=1.0 - specular_epsilon)
    return f0 + (f90 - f0) * (1.0 - cosTheta) ** 5.0


def bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08):
    _alpha = torch.clamp(alpha, min=min_roughness * min_roughness, max=1.0)
    alphaSqr = _alpha * _alpha

    h = _safe_normalize(wo + wi)
    woDotN = _dot(wo, nrm)
    wiDotN = _dot(wi, nrm)
    woDotH = _dot(wo, h)
    nDotH = _dot(nrm, h)

    D = bsdf_ndf_ggx(alphaSqr, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alphaSqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(col, 1, woDotH)

    w = F * D * G * 0.25 / torch.clamp(woDotN, min=specular_epsilon)

    frontfacing = (woDotN > specular_epsilon) & (wiDotN > specular_epsilon)
    return torch.where(frontfacing, w, torch.zeros_like(w))


def bsdf_lambert(nrm, wi):
    return torch.clamp(_dot(nrm, wi), min=0.0) / math.pi


def bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08):
    wo = _safe_normalize(view_pos - pos)
    wi = _safe_normalize(light_pos - pos)

    spec_str = arm[..., 0:1]  # x component
    roughness = arm[..., 1:2]  # y component
    metallic = arm[..., 2:3]  # z component
    ks = (0.04 * (1.0 - metallic) + kd * metallic) * (1 - spec_str)
    kd = kd * (1.0 - metallic)

    diffuse = kd * bsdf_lambert(nrm, wi)
    specular = bsdf_pbr_specular(ks, nrm, wo, wi, roughness * roughness, min_roughness=min_roughness)
    return kd + specular


def _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl):
    smooth_bitang = _safe_normalize(torch.cross(smooth_tng, smooth_nrm))
    if opengl:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] - smooth_bitang * perturbed_nrm[...,
                                                                             1:2] + smooth_nrm * torch.clamp(
            perturbed_nrm[..., 2:3], min=0.0)
    else:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] + smooth_bitang * perturbed_nrm[...,
                                                                             1:2] + smooth_nrm * torch.clamp(
            perturbed_nrm[..., 2:3], min=0.0)
    return _safe_normalize(shading_nrm)


def _bend_normal(view_vec, smooth_nrm, geom_nrm, two_sided_shading):
    # Swap normal direction for backfacing surfaces
    if two_sided_shading:
        smooth_nrm = torch.where(_dot(geom_nrm, view_vec) > 0, smooth_nrm, -smooth_nrm)
        geom_nrm = torch.where(_dot(geom_nrm, view_vec) > 0, geom_nrm, -geom_nrm)

    t = torch.clamp(_dot(view_vec, smooth_nrm) / 0.1, min=0, max=1)
    return torch.lerp(geom_nrm, smooth_nrm, t)


def bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading,
                                opengl):
    """
    blend geometric normal and lerp normal
    :param pos: 
    :param view_pos: 
    :param perturbed_nrm: 
    :param smooth_nrm: 
    :param smooth_tng: 
    :param geom_nrm: 
    :param two_sided_shading: 
    :param opengl: 
    :return: 
    """
    smooth_nrm = _safe_normalize(smooth_nrm)
    smooth_tng = _safe_normalize(smooth_tng)
    view_vec = _safe_normalize(view_pos - pos)
    shading_nrm = _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl)
    return _bend_normal(view_vec, shading_nrm, geom_nrm, two_sided_shading)


 
   
   
   
   
   
   
   
   
   