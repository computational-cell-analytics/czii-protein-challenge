import torch


class FocalLossWithLabelSmoothing(torch.nn.Module):
    """
    Combines Focal Loss with Label Smoothing for multi-class classification.
    Args:
        num_classes: number of classes
        gamma: focal loss focusing parameter (default 2.0)
        alpha: class weighting factor (float or tensor)
        label_smoothing: smoothing value epsilon (0 = no smoothing)
    """

    def __init__(self, num_classes, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        if alpha is None:
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, logits, target):
        """
        logits: (N, C)
        target: (N,) integer class labels
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) 

        #Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)

        #Focal loss
        probs = log_probs.exp()               # p = softmax(x)
        focal_weight = (1 - probs) ** self.gamma

        # class-weighting factor alpha
        alpha_factor = self.alpha.to(logits.device)[target]
        alpha_factor = alpha_factor.unsqueeze(1)  # shape (N,1)

        # combine all terms
        loss = -true_dist * alpha_factor * focal_weight * log_probs

        return loss.sum(dim=1).mean()