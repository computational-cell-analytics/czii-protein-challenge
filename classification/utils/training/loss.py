import torch


class FocalLossWithLabelSmoothing(torch.nn.Module):
    """
    Combines Focal Loss with Label Smoothing for multi-class classification.

    Args:
        num_classes: number of classes
        gamma: focal loss focusing parameter (default 2.0)
        alpha: class weighting factor. One of:
            * None -> uniform weights (no class balancing)
            * "balanced" / "inverse_freq" -> inverse-frequency weights computed
              from the training targets at runtime via set_alpha_from_targets
              (no need to hard-code per-dataset class counts)
            * a float, sequence, or tensor of per-class weights
        alpha_beta: tempering exponent for the data-derived weights,
            alpha_c proportional to (1 / count_c) ** alpha_beta. 1.0 is full
            inverse-frequency; 0.5 is gentler sqrt-tempered weighting (recommended
            with focal gamma > 0 and strong imbalance); 0.0 is uniform. Only used
            when alpha is "balanced"/"inverse_freq".
        label_smoothing: smoothing value epsilon (0 = no smoothing)

    Note on the alpha convention: alpha weights each sample by the weight of its
    *true* class (alpha_t), matching the original Focal Loss paper and
    torch.nn.CrossEntropyLoss(weight=...). This is the standard way to correct
    class imbalance.
    """

    # alpha aliases that mean "derive inverse-frequency weights from the data"
    _AUTO_MODES = ("balanced", "inverse_freq")

    def __init__(self, num_classes, gamma=2.0, alpha=None, alpha_beta=1.0, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha_beta = alpha_beta
        self.label_smoothing = label_smoothing

        if isinstance(alpha, str):
            if alpha not in self._AUTO_MODES:
                raise ValueError(
                    f"Unknown alpha mode '{alpha}'. Use one of {self._AUTO_MODES}, "
                    "None, or a sequence of per-class weights."
                )
            # Filled in from the training targets later; uniform until then.
            self.alpha_mode = alpha
            self.register_buffer("alpha", torch.ones(num_classes) / num_classes)
        elif alpha is None:
            self.alpha_mode = None
            self.register_buffer("alpha", torch.ones(num_classes) / num_classes)
        else:
            self.alpha_mode = None
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.ndim == 0:
                alpha_tensor = alpha_tensor.repeat(num_classes)
            if alpha_tensor.numel() != num_classes:
                raise ValueError(
                    f"alpha has {alpha_tensor.numel()} entries but num_classes={num_classes}."
                )
            self.register_buffer("alpha", alpha_tensor)

    @property
    def needs_alpha(self) -> bool:
        """Whether alpha still has to be computed from the training data."""
        return self.alpha_mode in self._AUTO_MODES

    def set_alpha_from_targets(self, targets, label_to_index=None, beta=None):
        """Compute (optionally tempered) inverse-frequency class weights from training targets.

        Per class: alpha_c proportional to (1 / count_c) ** beta, then normalized
        so the count-weighted mean of alpha is 1 -- this keeps the overall loss
        scale comparable to uniform alpha regardless of beta. beta=1 is full
        inverse-frequency, beta=0.5 is gentler sqrt-tempering, beta=0 is uniform.
        Classes absent from the training set get weight 0.

        Args:
            targets: iterable of labels as stored by the dataset.
            label_to_index: optional mapping raw label -> integer class index,
                matching the integer targets the loss receives. If None, targets
                are assumed to already be integer class indices.
            beta: tempering exponent; defaults to self.alpha_beta.

        Returns:
            (counts, alpha) tensors of shape (num_classes,).
        """
        if beta is None:
            beta = self.alpha_beta

        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for t in targets:
            idx = label_to_index[t] if label_to_index is not None else int(t)
            counts[idx] += 1.0

        total = counts.sum()
        if total == 0:
            raise ValueError("Cannot compute inverse-frequency alpha from empty targets.")

        present = counts > 0
        raw = torch.zeros(self.num_classes, dtype=torch.float32)
        raw[present] = (1.0 / counts[present]) ** beta

        # Normalize to count-weighted mean 1, so loss scale ~ uniform for any beta.
        norm = (counts * raw).sum() / total
        alpha = raw / norm  # absent classes stay 0

        self.alpha = alpha.to(self.alpha.device)
        self.alpha_mode = None  # resolved
        return counts, alpha

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