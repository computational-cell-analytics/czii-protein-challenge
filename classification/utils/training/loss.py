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

        # Label smoothing: build the smoothed target distribution.
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)

        # Per-sample cross-entropy against the smoothed target.
        ce = -(true_dist * log_probs).sum(dim=1)  # (N,)

        # Focal modulation is a PER-SAMPLE scalar based on the probability of the
        # *true* class, (1 - p_t) ** gamma -- exactly as in the Focal Loss paper.
        # (Applying (1 - p) ** gamma per class inside the sum is wrong: the loss
        # then does not vanish for well-classified samples and, scaled by alpha,
        # actively penalizes confident correct predictions on up-weighted classes.)
        pt = log_probs.exp().gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)
        focal_weight = (1.0 - pt) ** self.gamma  # (N,)

        # Class-weighting factor alpha_t (weight of the true class).
        alpha_factor = self.alpha.to(logits.device)[target]  # (N,)

        loss = alpha_factor * focal_weight * ce  # (N,)
        return loss.mean()


class BalancedSoftmaxLoss(torch.nn.Module):
    """Balanced Softmax / logit-adjusted cross-entropy with optional label smoothing.

    During training the class log-priors are added to the logits before the
    softmax::

        adjusted_logit_j = logit_j + tau * log(prior_j)

    This makes the softmax Bayes-consistent for the *balanced* error rate
    (equivalently macro-averaged recall) rather than plain accuracy, which is why
    it tends to beat focal loss on mildly/moderately imbalanced data without any
    per-class weight tuning. With tau=1 and prior = training class frequency this
    is exactly the "Balanced Softmax" of Ren et al. (2020); tau tempers the
    strength of the correction (the "logit adjustment" of Menon et al. (2020)):
    tau=0 recovers plain (smoothed) cross-entropy, larger tau pushes harder
    toward the rare classes.

    IMPORTANT -- the log-prior term is a TRAINING-ONLY correction. At inference /
    validation you must score the RAW model logits (do NOT add the prior); the
    argmax of the raw logits is the Bayes-optimal balanced prediction. This loss
    must therefore never be applied at eval time. In this codebase the trainer
    computes its metric directly on the model output, so nothing extra is needed.

    Args:
        num_classes: number of classes.
        tau: tempering factor on the log-prior adjustment (default 1.0 = standard
            Balanced Softmax). 0.0 disables the adjustment (plain smoothed CE).
        prior: class prior. One of:
            * None / "balanced" / "data" -> class frequencies computed from the
              training targets at runtime via set_prior_from_targets (uniform,
              i.e. no adjustment, until then).
            * a sequence/tensor of per-class priors (need not be normalized;
              they are renormalized to sum to 1).
        label_smoothing: smoothing value epsilon (0 = no smoothing).
    """

    # prior aliases that mean "derive class frequencies from the data"
    _AUTO_MODES = ("balanced", "data")

    # numerical floor so absent classes get a very negative (but finite) log-prior
    _EPS = 1e-12

    def __init__(self, num_classes, tau=1.0, prior=None, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.label_smoothing = label_smoothing

        # log_prior is what actually enters the forward pass. Uniform prior has a
        # constant log_prior that cancels in the softmax -> reduces to plain CE,
        # which is a safe fallback before the data-derived prior is filled in.
        uniform_log_prior = torch.full((num_classes,), -float(torch.log(torch.tensor(float(num_classes)))))

        if prior is None or (isinstance(prior, str) and prior in self._AUTO_MODES):
            self.prior_mode = "balanced" if prior is None else prior
            self.register_buffer("log_prior", uniform_log_prior)
        elif isinstance(prior, str):
            raise ValueError(
                f"Unknown prior mode '{prior}'. Use one of {self._AUTO_MODES}, "
                "None, or a sequence of per-class priors."
            )
        else:
            self.prior_mode = None
            prior_tensor = torch.as_tensor(prior, dtype=torch.float32)
            if prior_tensor.numel() != num_classes:
                raise ValueError(
                    f"prior has {prior_tensor.numel()} entries but num_classes={num_classes}."
                )
            prior_tensor = prior_tensor / prior_tensor.sum()
            self.register_buffer("log_prior", prior_tensor.clamp_min(self._EPS).log())

    @property
    def needs_prior(self) -> bool:
        """Whether the prior still has to be computed from the training data."""
        return self.prior_mode in self._AUTO_MODES

    def set_prior_from_targets(self, targets, label_to_index=None):
        """Compute class-frequency priors from the training targets.

        Per class: prior_c = count_c / total. The (log) prior is stored for use in
        the forward pass. Classes absent from the training set get a tiny prior so
        their log-prior is very negative (they are effectively never predicted
        during training, which matches focal loss giving absent classes weight 0).

        Args:
            targets: iterable of labels as stored by the dataset.
            label_to_index: optional mapping raw label -> integer class index. If
                None, targets are assumed to already be integer class indices.

        Returns:
            (counts, prior) tensors of shape (num_classes,).
        """
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for t in targets:
            idx = label_to_index[t] if label_to_index is not None else int(t)
            counts[idx] += 1.0

        total = counts.sum()
        if total == 0:
            raise ValueError("Cannot compute a class-frequency prior from empty targets.")

        prior = counts / total
        self.log_prior = prior.clamp_min(self._EPS).log().to(self.log_prior.device)
        self.prior_mode = None  # resolved
        return counts, prior

    def forward(self, logits, target):
        """
        logits: (N, C)
        target: (N,) integer class labels
        """
        # Training-only logit adjustment: shift each class logit by tau*log(prior).
        adjusted = logits + self.tau * self.log_prior.to(logits.device)
        log_probs = torch.nn.functional.log_softmax(adjusted, dim=-1)

        # Label smoothing: build the smoothed target distribution (one-hot if eps=0).
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            if self.num_classes > 1:
                true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)

        loss = -(true_dist * log_probs).sum(dim=1)  # (N,)
        return loss.mean()