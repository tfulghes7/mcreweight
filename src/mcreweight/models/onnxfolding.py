from mcreweight.models.onnxreweighter import (
    BaseONNXReweighter,
    ONNXGBReweighter,
    ONNXIXGBReweighter,
    ONNXINNReweighter,
)
import numpy as np


class BaseONNXFoldingReweighter(BaseONNXReweighter):
    """
    Generic k-folding ensemble wrapper for ONNX reweighters.

    This trains K separate reweighters, each on a subset of (original,target),
    and predicts weights by averaging the K predictions.

    Notes:
      - We shuffle indices by default to avoid ordering bias.
      - Each fold model fits its own transformer (as defined by that model).
    """

    def __init__(
        self,
        base_cls,
        n_folds=5,
        shuffle=True,
        random_state=42,
        transform=None,
        verbosity=1,
        aggregation="weighted_geometric",
        **base_params,
    ):
        super().__init__(transform=transform, verbosity=verbosity)
        self.base_cls = base_cls
        self.n_folds = int(n_folds)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.aggregation = aggregation
        self.base_params = dict(base_params)
        self.models = []
        self.fold_metrics_ = []
        self.fold_weights_ = None
        self.fold_indices_ = []
        self._train_original_size = None

    def _child_base_params(self, fold_index):
        params = dict(self.base_params)
        params.pop("transform", None)
        params.pop("verbosity", None)
        params["random_state"] = self.random_state + fold_index
        return params

    @staticmethod
    def _weighted_ks_1d(x1, x2, w1, w2):
        x1 = np.asarray(x1, dtype=np.float32)
        x2 = np.asarray(x2, dtype=np.float32)
        w1 = np.asarray(w1, dtype=np.float32)
        w2 = np.asarray(w2, dtype=np.float32)

        idx1 = np.argsort(x1)
        idx2 = np.argsort(x2)
        x1 = x1[idx1]
        x2 = x2[idx2]
        w1 = w1[idx1]
        w2 = w2[idx2]

        w1 = w1 / max(float(w1.sum()), 1e-12)
        w2 = w2 / max(float(w2.sum()), 1e-12)

        x_all = np.concatenate([x1, x2])
        x_all.sort()

        cdf1_idx = np.searchsorted(x1, x_all, side="right")
        cdf2_idx = np.searchsorted(x2, x_all, side="right")
        cdf1 = np.concatenate([[0.0], np.cumsum(w1, dtype=np.float64)])[cdf1_idx]
        cdf2 = np.concatenate([[0.0], np.cumsum(w2, dtype=np.float64)])[cdf2_idx]
        return float(np.max(np.abs(cdf1 - cdf2)))

    def _fold_metric(self, model, Xo_val, Xt_val, ow_val, tw_val):
        w_val = model.predict_weights(Xo_val, ow=ow_val)
        ks_vals = [
            self._weighted_ks_1d(Xo_val[:, j], Xt_val[:, j], w_val, tw_val)
            for j in range(Xo_val.shape[1])
        ]
        return float(np.mean(ks_vals))

    def _compute_fold_weights(self):
        metrics = np.asarray(self.fold_metrics_, dtype=np.float64)
        if metrics.size == 0:
            return np.ones(len(self.models), dtype=np.float64) / max(
                len(self.models), 1
            )

        raw = 1.0 / np.clip(metrics, 1e-6, np.inf)
        raw_sum = float(raw.sum())
        if raw_sum <= 0:
            return np.ones_like(raw) / len(raw)
        return raw / raw_sum

    def _make_folds(self, n):
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(idx)
        folds = np.array_split(idx, self.n_folds)
        return folds

    def fit(self, original, target, ow=None, tw=None):
        Xo = np.asarray(original)
        Xt = np.asarray(target)

        ow = None if ow is None else np.asarray(ow)
        tw = None if tw is None else np.asarray(tw)

        if self.n_folds < 2:
            raise ValueError("[FoldingReweighter] n_folds must be >= 2")

        self.models = []
        self.fold_metrics_ = []
        self.fold_indices_ = []
        self._train_original_size = len(Xo)

        folds_o = self._make_folds(len(Xo))
        folds_t = self._make_folds(len(Xt))

        for k in range(self.n_folds):
            # train on all but fold k
            train_idx_o = np.concatenate(
                [folds_o[i] for i in range(self.n_folds) if i != k]
            )
            train_idx_t = np.concatenate(
                [folds_t[i] for i in range(self.n_folds) if i != k]
            )
            val_idx_o = folds_o[k]
            val_idx_t = folds_t[k]

            m = self.base_cls(
                transform=self.transform,
                verbosity=self.verbosity,
                **self._child_base_params(k),
            )

            m.fit(
                Xo[train_idx_o],
                Xt[train_idx_t],
                None if ow is None else ow[train_idx_o],
                None if tw is None else tw[train_idx_t],
            )

            metric = self._fold_metric(
                m,
                Xo[val_idx_o],
                Xt[val_idx_t],
                (
                    np.ones(len(val_idx_o), dtype=np.float32)
                    if ow is None
                    else ow[val_idx_o]
                ),
                (
                    np.ones(len(val_idx_t), dtype=np.float32)
                    if tw is None
                    else tw[val_idx_t]
                ),
            )

            if self.verbosity >= 2:
                print(
                    f"[{self.__class__.__name__}] Fold {k+1}/{self.n_folds} trained. "
                    f"Validation mean KS={metric:.4f}"
                )

            self.models.append(m)
            self.fold_metrics_.append(metric)
            self.fold_indices_.append(np.asarray(val_idx_o, dtype=np.int64))

        self.fold_weights_ = self._compute_fold_weights()
        self.n_features_ = self.models[0].n_features_
        return self

    def predict_weights(self, X, ow=None):
        if not self.models:
            raise RuntimeError(f"[{self.__class__.__name__}] Model not fitted/loaded.")

        X = np.asarray(X)
        ow = None if ow is None else np.asarray(ow)

        w = np.vstack([m.predict_weights(X, ow=ow) for m in self.models]).astype(
            np.float64
        )

        if self.aggregation == "weighted_geometric" and self.fold_weights_ is not None:
            logw = np.log(np.clip(w, 1e-12, np.inf))
            w = np.exp(np.average(logw, axis=0, weights=self.fold_weights_))
        elif self.aggregation == "median":
            w = np.median(w, axis=0)
        else:
            w = np.exp(np.mean(np.log(np.clip(w, 1e-12, np.inf)), axis=0))

        if self.verbosity >= 3:
            print(
                f"[{self.__class__.__name__}] Predicted weights: "
                f"mean={w.mean():.3f}, std={w.std():.3f}, "
                f"min={w.min():.3f}, max={w.max():.3f}"
            )

        return w

    def predict_oof_weights(self, X, ow=None):
        if not self.models or not self.fold_indices_:
            raise RuntimeError(f"[{self.__class__.__name__}] Model not fitted/loaded.")

        X = np.asarray(X)
        if len(X) != self._train_original_size:
            return self.predict_weights(X, ow=ow)

        ow = np.ones(len(X), dtype=np.float32) if ow is None else np.asarray(ow)
        out = np.zeros(len(X), dtype=np.float64)
        for model, idx in zip(self.models, self.fold_indices_):
            out[idx] = model.predict_weights(X[idx], ow=ow[idx])
        return out.astype(np.float32)

    def save(self, prefix):
        # Save fold models with suffixes
        meta = {
            "n_folds": self.n_folds,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "transform": self.transform,
            "verbosity": self.verbosity,
            "aggregation": self.aggregation,
            "base_params": self.base_params,
            "fold_metrics": list(self.fold_metrics_),
            "fold_weights": (
                None if self.fold_weights_ is None else self.fold_weights_.tolist()
            ),
            "fold_indices": [idx.tolist() for idx in self.fold_indices_],
            "train_original_size": self._train_original_size,
        }
        self._export_meta(prefix + "_meta.pkl", meta)

        for i, m in enumerate(self.models):
            m.save(f"{prefix}_{i}")

    def load(self, prefix):
        meta = self._load_meta(prefix + "_meta.pkl")
        self.n_folds = meta["n_folds"]
        self.shuffle = meta["shuffle"]
        self.random_state = meta["random_state"]
        self.transform = meta["transform"]
        self.verbosity = meta["verbosity"]
        self.aggregation = meta.get("aggregation", "weighted_geometric")
        self.base_params = meta["base_params"]
        self.fold_metrics_ = list(meta.get("fold_metrics", []))
        self.fold_weights_ = meta.get("fold_weights", None)
        if self.fold_weights_ is not None:
            self.fold_weights_ = np.asarray(self.fold_weights_, dtype=np.float64)
        self.fold_indices_ = [
            np.asarray(idx, dtype=np.int64) for idx in meta.get("fold_indices", [])
        ]
        self._train_original_size = meta.get("train_original_size", None)

        self.models = []
        for i in range(self.n_folds):
            m = self.base_cls(
                transform=self.transform,
                verbosity=self.verbosity,
                **self._child_base_params(i),
            )
            m.load(f"{prefix}_{i}")
            self.models.append(m)

        self.n_features_ = self.models[0].n_features_


class ONNXIXGBFoldingReweighter(BaseONNXFoldingReweighter):
    """
    k-folding ensemble of ONNXIXGBReweighter models.
    """

    def __init__(
        self,
        n_folds=5,
        shuffle=True,
        random_state=42,
        transform=None,
        verbosity=1,
        **xgb_params,
    ):
        super().__init__(
            base_cls=ONNXIXGBReweighter,
            n_folds=n_folds,
            shuffle=shuffle,
            random_state=random_state,
            transform=transform,
            verbosity=verbosity,
            **xgb_params,
        )


class ONNXFoldingReweighter(BaseONNXFoldingReweighter):
    """
    k-folding ensemble of ONNXGBReweighter models.
    """

    def __init__(
        self,
        n_folds=5,
        shuffle=True,
        random_state=42,
        transform=None,
        verbosity=1,
        **gb_params,
    ):
        super().__init__(
            base_cls=ONNXGBReweighter,
            n_folds=n_folds,
            shuffle=shuffle,
            random_state=random_state,
            transform=transform,
            verbosity=verbosity,
            **gb_params,
        )


class ONNXINNFoldingReweighter(BaseONNXFoldingReweighter):
    """
    k-folding ensemble of ONNXINNReweighter models.
    """

    def __init__(
        self,
        n_folds=5,
        shuffle=True,
        random_state=42,
        transform=None,
        verbosity=1,
        **nn_params,
    ):
        super().__init__(
            base_cls=ONNXINNReweighter,
            n_folds=n_folds,
            shuffle=shuffle,
            random_state=random_state,
            transform=transform,
            verbosity=verbosity,
            **nn_params,
        )
