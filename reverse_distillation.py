"""Notebook adapter for RD4AD's ResNet18 reverse-distillation architecture.

Architecture: https://github.com/hq-deng/RD4AD (vendored under third_party/rd4ad).
Inputs must be ImageNet-normalized RGB tensors. Only normal training data is used.
"""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

from third_party.rd4ad.resnet import resnet18
from third_party.rd4ad.de_resnet import de_resnet18


class StudentMethod:
    def __init__(self, device=None, epochs=30, lr=0.005, seed=42,
                 score_hw=(28, 28), smooth_sigma=2.0, category="unknown"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.score_hw, self.smooth_sigma = tuple(score_hw), smooth_sigma
        self.category = category
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.teacher, self.bottleneck = resnet18(pretrained=True)
        self.decoder = de_resnet18(pretrained=False)
        self.teacher.requires_grad_(False).eval().to(self.device)
        self.bottleneck.to(self.device)
        self.decoder.to(self.device)
        self.history = []
        self.is_fitted = False
        self.fit_runtime_sec = 0.0

    @staticmethod
    def loss(targets, predictions):
        # Original RD4AD objective: cosine distance of flattened feature levels.
        return sum((1 - F.cosine_similarity(t.flatten(1), p.flatten(1), dim=1)).mean()
                   for t, p in zip(targets, predictions))

    @torch.no_grad()
    def _cache_normal_features(self, loader):
        chunks = [[], [], []]
        for batch in loader:
            if torch.any(batch["label"] != 0):
                raise ValueError("Reverse Distillation fit() accepts train/good only.")
            features = self.teacher(batch["image"].to(self.device))
            for parts, feature in zip(chunks, features):
                parts.append(feature.cpu().to(torch.float16))
        if not chunks[0]:
            raise ValueError("The normal training set is empty.")
        return [torch.cat(parts) for parts in chunks]

    def fit(self, train_loader):
        self.is_fitted = False
        start = time.perf_counter()
        self.teacher.eval()
        features = self._cache_normal_features(train_loader)
        n = features[0].shape[0]
        if n < 2:
            raise ValueError("At least two normal images are required for BatchNorm training.")
        batch_size = min(train_loader.batch_size or 16, n)
        # Avoid singleton batches without dropping any training images.
        batches = max(1, (n + batch_size - 1) // batch_size)
        if n // batches < 2:
            batches -= 1
        generator = torch.Generator().manual_seed(self.seed)
        params = list(self.bottleneck.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.999))
        print(f"{self.category}: cached {n} normal-image feature pyramids", flush=True)
        self.history = []
        for epoch in range(self.epochs):
            self.bottleneck.train()
            self.decoder.train()
            permutation = torch.randperm(n, generator=generator)
            total = 0.0
            for indices in permutation.tensor_split(batches):
                targets = [level[indices].to(self.device, dtype=torch.float32) for level in features]
                predictions = self.decoder(self.bottleneck(targets))
                loss = self.loss(targets, predictions)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite RD loss at epoch {epoch + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
            self.history.append(total / n)
            print(f"{self.category}: epoch {epoch + 1}/{self.epochs}, loss={total / n:.6f}", flush=True)
        self.fit_runtime_sec = time.perf_counter() - start
        self.is_fitted = True

    @torch.no_grad()
    def predict(self, test_loader):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        start = time.perf_counter()
        self.teacher.eval()
        self.bottleneck.eval()
        self.decoder.eval()
        paths, maps = [], []
        for batch in test_loader:
            targets = self.teacher(batch["image"].to(self.device))
            predictions = self.decoder(self.bottleneck(targets))
            # Local channel-wise cosine discrepancies from all three scales.
            combined = sum(F.interpolate(
                (1 - F.cosine_similarity(t, p, dim=1)).clamp_min(0).unsqueeze(1),
                size=self.score_hw, mode="bilinear", align_corners=False
            ) for t, p in zip(targets, predictions))
            for anomaly_map in combined[:, 0].cpu().numpy():
                maps.append(gaussian_filter(anomaly_map, self.smooth_sigma))
            paths.extend(str(p) for p in batch["image_path"])
        score_maps = np.asarray(maps, dtype=np.float32).reshape(-1, *self.score_hw)
        if not np.isfinite(score_maps).all():
            raise FloatingPointError("Non-finite anomaly maps.")
        # Raw scores retain a scale usable with a later normal-validation threshold.
        scores = score_maps.max(axis=(1, 2)) if len(maps) else np.empty(0, dtype=np.float32)
        return dict(paths=paths, scores=scores.tolist(), score_maps=score_maps,
                    runtime_sec=time.perf_counter() - start,
                    fit_runtime_sec=self.fit_runtime_sec, history=list(self.history),
                    method="ReverseDistillation-ResNet18")

    def save(self, path):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            teacher=self.teacher.state_dict(), bottleneck=self.bottleneck.state_dict(),
            decoder=self.decoder.state_dict(), history=self.history,
            config=dict(epochs=self.epochs, lr=self.lr, seed=self.seed,
                        score_hw=self.score_hw, smooth_sigma=self.smooth_sigma,
                        category=self.category),
        ), path)

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        detector = cls(device=device, **checkpoint["config"])
        detector.teacher.load_state_dict(checkpoint["teacher"])
        detector.bottleneck.load_state_dict(checkpoint["bottleneck"])
        detector.decoder.load_state_dict(checkpoint["decoder"])
        detector.history = checkpoint["history"]
        detector.is_fitted = True
        return detector
