import torch


# ACC
def calc_anomaly_correlation_coefficient(x,y):
    # r"""Pearson product-moment correlation coefficient.

    # A measure of the linear association between the forecast and verification data that
    # is independent of the mean and variance of the individual distributions. This is
    # also known as the Anomaly Correlation Coefficient (ACC) when correlating anomalies.

    # .. math::
    #     corr = \frac{cov(f, o)}{\sigma_{f}\cdot\sigma_{o}},

    # where :math:`\sigma_{f}` and :math:`\sigma_{o}` represent the standard deviation
    # of the forecast and verification data over the experimental period, respectively.
    
    # Args:

    # """
    #return torch.cov(torch.cat((x.unsqueeze(1),y.unsqueeze(1)), dim=1)) / (torch.std(x,dim=0, unbiased=True)*torch.std(y,dim=0, unbiased=True))
    
    #ACC computation without m
    # https://confluence.ecmwf.int/display/FUG/Section+12.A+Statistical+Concepts+-+Deterministic+Data
    num = (x*y).mean()
    square_denom = (x**2).mean()*(y**2).mean()
    # averageing on batch samples
    res = torch.mean(num / torch.sqrt(square_denom), dim=0)
    return res