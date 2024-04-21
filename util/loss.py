import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, weight=None):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(WeightedLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.weights = weight

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)

        if self.weights is not None:
            logprobs = logprobs * self.weights

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        if self.smoothing > 0.:
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        else:
            loss = nll_loss

        if self.weights is not None:
            wt = self.weights.gather(dim=-1, index=target)
            return loss.sum() / wt.sum()
        else:
            return loss.mean()
