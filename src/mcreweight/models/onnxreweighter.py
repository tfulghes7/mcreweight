import numpy as np
import os
import joblib
import copy
import inspect

from typing import Optional, Dict, Any

## Scikit-learn imports
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler

## ONNX imports
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost


class BaseONNXReweighter:
    """
    Abstract base class for ONNX-based reweighters.

    Provides:
      - consistent float32 / 2D shape checks
      - weight checks + normalization helpers
      - stable logit/margin utilities for density-ratio updates
      - feature transformations with fit-on-mixture pattern
      - ONNX session loading + robust proba extraction
      - meta save/load helpers
    """

    def __init__(
        self,
        transform: Optional[str] = None,
        eps: float = 1e-6,
        verbosity: int = 1,
        random_state: int = 42,
        weight_norm: str = "mean1",  # "mean1" or "sum1"
    ):
        self.onnx_session = None
        self.n_features_ = None

        # Transformations
        self.transform = transform
        self.qt = None
        self.pt = None
        self.scaler = None
        self._transform_is_fitted = False

        # Numerics / logging
        self.eps = float(eps)
        self.verbosity = int(verbosity) if verbosity is not None else 1
        self.random_state = int(random_state)

        # Normalization policy for mixed-class weights
        if weight_norm not in ("mean1", "sum1"):
            raise ValueError(
                "[BaseONNXReweighter] weight_norm must be 'mean1' or 'sum1'"
            )
        self.weight_norm = weight_norm
        self.require_positive_ow_for_fit = False
        self.require_positive_tw_for_fit = False

    # -------------------------
    # Public API (abstract)
    # -------------------------
    def fit(self, original, target, ow=None, tw=None):
        raise NotImplementedError

    def predict_weights(self, X, ow=None):
        raise NotImplementedError

    def save(self, prefix: str):
        raise NotImplementedError

    def load(self, prefix: str):
        raise NotImplementedError

    # -------------------------
    # RNG
    # -------------------------
    def _rng(self):
        return np.random.RandomState(self.random_state)

    # -------------------------
    # Shape / type utilities
    # -------------------------
    def _ensure_2d_float32(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(
                f"[BaseONNXReweighter] Expected 2D input, got shape {X.shape}"
            )
        if self.n_features_ is None:
            self.n_features_ = int(X.shape[1])
        elif int(X.shape[1]) != int(self.n_features_):
            raise ValueError(
                f"[BaseONNXReweighter] Expected {self.n_features_} features, got {X.shape[1]}"
            )
        return X

    def _make_positive_weights(self, w: np.ndarray, context: str) -> np.ndarray:
        w = np.asarray(w, dtype=np.float32).copy()
        neg = w < 0
        if np.any(neg):
            if self.verbosity >= 1:
                print(
                    f"[BaseONNXReweighter] {context}: clipping {int(np.count_nonzero(neg))}/"
                    f"{len(w)} negative weights to 0 for estimator compatibility."
                )
            w[neg] = 0.0
        if not np.any(w > 0):
            raise ValueError(
                f"[BaseONNXReweighter] {context}: no positive weights remain after clipping."
            )
        return w.astype(np.float32)

    def _ensure_weights(
        self,
        X: np.ndarray,
        w: Optional[np.ndarray],
        allow_negative: bool = True,
        context: str = "weights",
    ) -> np.ndarray:
        n = len(X)
        if w is None:
            w = np.ones(n, dtype=np.float32)
        else:
            w = np.asarray(w, dtype=np.float32)
            if w.ndim != 1 or len(w) != n:
                raise ValueError(
                    "[BaseONNXReweighter] weights must be 1D and match number of samples"
                )
            if not np.all(np.isfinite(w)):
                raise ValueError(f"[BaseONNXReweighter] {context} must be finite.")

        if allow_negative:
            return w.astype(np.float32)
        return self._make_positive_weights(w, context)

    # -------------------------
    # Weight normalization
    # -------------------------
    def _normalize_by_class(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Normalize weights per class.

        - First: for each class c: sum_{i in c} w_i = 1
        - Then: scale globally according to weight_norm:
            "mean1": mean(w) = 1  (preserves typical reweighter conventions)
            "sum1":  sum(w)  = 1  (useful for density-style histograms / ratios)
        """
        y = np.asarray(y)
        w = np.asarray(w, dtype=np.float32).copy()

        for cls in np.unique(y):
            m = y == cls
            s = float(w[m].sum())
            if s > 0:
                w[m] /= s

        if self.weight_norm == "mean1":
            mu = float(w.mean())
            if mu > 0:
                w /= mu
        else:  # "sum1"
            s = float(w.sum())
            if s > 0:
                w /= s

        return w.astype(np.float32)

    # -------------------------
    # Stable logit/margin utilities
    # -------------------------
    def _sigmoid(self, m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float32)
        # stable sigmoid
        out = np.empty_like(m, dtype=np.float32)
        pos = m >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-m[pos]))
        expm = np.exp(m[~pos])
        out[~pos] = expm / (1.0 + expm)
        return out

    def _logit(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float32), self.eps, 1.0 - self.eps)
        return (np.log(p) - np.log1p(-p)).astype(np.float32)

    def _delta_from_proba(self, p: np.ndarray) -> np.ndarray:
        """
        Your original delta:
            delta = log((1-p)/p) = -logit(p)
        Prefer margin-based delta when possible.
        """
        return (-self._logit(p)).astype(np.float32)

    # -------------------------
    # Transformations
    # -------------------------
    def fit_transform_on_mixture(self, Xo: np.ndarray, Xt: np.ndarray):
        """
        Fit the feature transform on a representative mixture once.
        """
        X_mix = np.vstack([Xo, Xt]).astype(np.float32)
        self._transform_fit(X_mix)

    def _transform_fit(self, X: np.ndarray):
        X = self._ensure_2d_float32(X)

        if self.transform is None:
            self._transform_is_fitted = True
            return

        t = str(self.transform).lower()
        if "quantile" in t:
            # cap n_quantiles to n_samples for stability
            n_q = min(10_000, max(10, X.shape[0]))
            self.qt = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution="uniform",
                random_state=self.random_state,
                subsample=int(1e6),  # sklearn default is 1e5; you can tune
            )
            self.qt.fit(X)
        elif "yeo" in t:
            self.pt = PowerTransformer(method="yeo-johnson", standardize=True)
            self.pt.fit(X)
        elif "log" in t:
            pass
        elif "scaler" in t:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        else:
            raise ValueError(
                f"[BaseONNXReweighter] Unsupported transformation: {self.transform}"
            )

        self._transform_is_fitted = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self._transform_is_fitted:
            raise RuntimeError(
                "[BaseONNXReweighter] Transformation not fitted. Fit first."
            )

        X = self._ensure_2d_float32(X)

        if self.transform is None:
            return X

        t = str(self.transform).lower()
        if "quantile" in t:
            return self.qt.transform(X).astype(np.float32)
        elif "yeo" in t:
            return self.pt.transform(X).astype(np.float32)
        elif "log" in t:
            return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)
        elif "scaler" in t:
            return self.scaler.transform(X).astype(np.float32)
        else:
            raise ValueError(
                f"[BaseONNXReweighter] Unsupported transformation: {self.transform}"
            )

    # -------------------------
    # ONNX I/O
    # -------------------------
    def _load_onnx_session(self, path: str):
        self.onnx_session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        return self.onnx_session

    def _export_meta(self, path: str, meta: Dict[str, Any]):
        joblib.dump(meta, path)

    def _load_meta(self, path: str) -> Dict[str, Any]:
        return joblib.load(path)

    @staticmethod
    def _onnx_get_input_name(session: ort.InferenceSession) -> str:
        ins = session.get_inputs()
        if not ins:
            raise RuntimeError("ONNX session has no inputs")
        # prefer the first
        return ins[0].name

    @staticmethod
    def _onnx_get_class1_proba(
        session: ort.InferenceSession, X: np.ndarray
    ) -> np.ndarray:
        """
        Robustly pull P(class=1) from common sklearn/xgb ONNX exports.
        """
        X = np.asarray(X, dtype=np.float32)

        input_name = BaseONNXReweighter._onnx_get_input_name(session)
        outs = session.run(None, {input_name: X})

        out_infos = session.get_outputs()
        names = [o.name.lower() for o in out_infos]

        prob = None
        for i, n in enumerate(names):
            if "prob" in n or "proba" in n:
                prob = np.asarray(outs[i])
                break
        if prob is None:
            for out in outs:
                a = np.asarray(out)
                if a.dtype.kind == "f":
                    prob = a
                    break
        if prob is None:
            raise RuntimeError(
                f"Could not find probability output. outs={[(np.asarray(o).shape, np.asarray(o).dtype) for o in outs]}"
            )

        if prob.ndim == 2 and prob.shape[1] == 2:
            p = prob[:, 1]
        elif prob.ndim == 1:
            p = prob
        else:
            raise RuntimeError(f"Unexpected prob shape: {prob.shape}")

        return np.clip(p.astype(np.float32), 1e-6, 1.0 - 1e-6)

    @staticmethod
    def _onnx_get_regression_output(
        session: ort.InferenceSession, X: np.ndarray
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        input_name = BaseONNXReweighter._onnx_get_input_name(session)
        outs = session.run(None, {input_name: X})

        reg = None
        for out in outs:
            arr = np.asarray(out)
            if arr.dtype.kind == "f":
                reg = arr
                break
        if reg is None:
            raise RuntimeError("Could not find regression output in ONNX session.")

        reg = np.asarray(reg, dtype=np.float32)
        if reg.ndim == 2 and reg.shape[1] == 1:
            reg = reg[:, 0]
        return reg.astype(np.float32)


class ONNXGBReweighter(BaseONNXReweighter):
    """
    ONNX-exportable implementation of hep_ml's GBReweighter logic.

    This mirrors the signed-weight handling of hep_ml's ReweightLossFunction:
      - tree targets are class/sign-based residuals
      - tree fit weights use abs(normalized signed weights)
      - leaf updates are log(target_weight + reg) - log(original_weight + reg)
    """

    def __init__(
        self,
        transform=None,
        verbosity=1,
        n_estimators=40,
        learning_rate=0.2,
        max_depth=3,
        min_samples_leaf=200,
        loss_regularization=5.0,
        subsample=1.0,
        min_samples_split=2,
        max_features=None,
        max_leaf_nodes=None,
        splitter="best",
        update_tree=True,
        random_state=42,
        store_dir=None,
        eps=1e-6,
    ):
        super().__init__(
            transform=transform, eps=eps, verbosity=verbosity, random_state=random_state
        )
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.loss_regularization = float(loss_regularization)
        self.subsample = float(subsample)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.splitter = splitter
        self.update_tree = bool(update_tree)
        self.store_dir = store_dir

        self.initial_step_ = 0.0
        self.stage_estimators_ = []
        self.stage_paths_ = []
        self.stage_sessions_ = []
        self.sample_weight_ = None
        self.y_ = None
        self.signs_ = None
        self.mask_original_ = None
        self.mask_target_ = None

    def _check_params(self):
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not (0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")

    def _normalize_signed_weights(self, y, sample_weight):
        w = np.asarray(sample_weight, dtype=np.float64).copy()
        for value in np.unique(y):
            mask = y == value
            denom = np.sum(w[mask])
            if denom != 0:
                w[mask] /= denom
        mean = np.mean(w)
        if mean != 0:
            w /= mean
        return w.astype(np.float64)

    def _compute_weights(self, y_pred):
        weights = self.sample_weight_ * np.exp(self.y_ * y_pred)
        return self._normalize_signed_weights(self.y_, weights)

    def _make_stage_tree(self):
        return DecisionTreeRegressor(
            criterion="squared_error",
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
        )

    @staticmethod
    def _apply_leaf_values(tree, leaf_values):
        tree_copy = copy.deepcopy(tree)
        values = tree_copy.tree_.value
        if values.ndim == 3:
            values[:, 0, 0] = leaf_values
        else:
            values[:, 0] = leaf_values
        return tree_copy

    def _estimate_tree(self, tree, X):
        return tree.predict(X).astype(np.float64)

    def get_params(self):
        return {
            "transform": self.transform,
            "verbosity": self.verbosity,
            "random_state": self.random_state,
            "eps": self.eps,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "loss_regularization": self.loss_regularization,
            "subsample": self.subsample,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "splitter": self.splitter,
            "update_tree": self.update_tree,
            "store_dir": self.store_dir,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def fit(self, original, target, ow=None, tw=None):
        self._check_params()
        self.n_features_ = None

        Xo = self._ensure_2d_float32(original)
        Xt = self._ensure_2d_float32(target)
        ow = self._ensure_weights(
            Xo, ow, allow_negative=True, context="original training weights"
        ).astype(np.float64)
        tw = self._ensure_weights(
            Xt, tw, allow_negative=True, context="target training weights"
        ).astype(np.float64)

        X_mix = np.vstack([Xo, Xt]).astype(np.float32)
        self._transform_fit(X_mix)
        Xo = self._transform(Xo)
        Xt = self._transform(Xt)

        X = np.vstack([Xo, Xt]).astype(np.float32)
        y = np.concatenate(
            [np.ones(len(Xo), dtype=np.int32), np.zeros(len(Xt), dtype=np.int32)]
        )
        sample_weight = np.concatenate([ow, tw]).astype(np.float64)

        self.sample_weight_ = sample_weight
        self.y_ = y
        self.signs_ = (2 * y - 1) * np.sign(sample_weight)
        self.mask_original_ = y.astype(np.float64)
        self.mask_target_ = (1 - y).astype(np.float64)

        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        n_inbag = max(1, int(self.subsample * n_samples))

        y_pred = np.zeros(n_samples, dtype=np.float64) + self.initial_step_
        self.stage_estimators_.clear()
        self.stage_paths_.clear()
        self.stage_sessions_.clear()

        for stage in range(self.n_estimators):
            tree = self._make_stage_tree()
            norm_weights = self._compute_weights(y_pred)
            residual = self.signs_
            fit_weights = np.abs(norm_weights)
            train_idx = rng.choice(n_samples, size=n_inbag, replace=False)

            tree.fit(
                X[train_idx], residual[train_idx], sample_weight=fit_weights[train_idx]
            )
            terminal_regions = tree.apply(X)
            leaf_values = tree.tree_.value[:, 0, 0].copy()

            if self.update_tree:
                w_target = np.bincount(
                    terminal_regions,
                    weights=self.mask_target_ * norm_weights,
                    minlength=len(leaf_values),
                )
                w_original = np.bincount(
                    terminal_regions,
                    weights=self.mask_original_ * norm_weights,
                    minlength=len(leaf_values),
                )
                w_target = np.clip(w_target, 0.0, np.inf)
                w_original = np.clip(w_original, 0.0, np.inf)
                leaf_values = np.log(w_target + self.loss_regularization) - np.log(
                    w_original + self.loss_regularization
                )

            tree = self._apply_leaf_values(tree, leaf_values)
            y_pred += self.learning_rate * self._estimate_tree(tree, X)
            self.stage_estimators_.append(tree)

            if self.verbosity >= 3:
                print(
                    f"[ONNXGBReweighter] stage {stage+1}/{self.n_estimators} "
                    f"pred mean={np.mean(y_pred):.4f} std={np.std(y_pred):.4f}"
                )

        return self

    def predict_weights(self, X, ow=None):
        X = self._ensure_2d_float32(X)
        ow = self._ensure_weights(
            X, ow, allow_negative=True, context="prediction weights"
        ).astype(np.float64)
        X = self._transform(X)

        score = np.zeros(len(X), dtype=np.float64) + self.initial_step_
        self._ensure_sessions_loaded()
        if self.stage_sessions_:
            for sess in self.stage_sessions_:
                score += self.learning_rate * self._onnx_get_regression_output(sess, X)
        elif self.stage_estimators_:
            for tree in self.stage_estimators_:
                score += self.learning_rate * self._estimate_tree(tree, X)
        else:
            raise RuntimeError(
                "[ONNXGBReweighter] No stages available. Fit or load the model first."
            )

        multipliers = np.exp(score)
        return (multipliers * ow).astype(np.float32)

    def _ensure_sessions_loaded(self):
        if self.stage_sessions_ or not self.stage_paths_:
            return
        self.stage_sessions_ = [self._load_onnx_session(p) for p in self.stage_paths_]

    def _export_estimator_to_onnx(self, estimator, n_features, path):
        initial = [("input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(estimator, initial_types=initial, target_opset=17)
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())

    def save(self, prefix):
        stages_dir = prefix + "_stages"
        os.makedirs(stages_dir, exist_ok=True)

        self.stage_paths_ = []
        for i, tree in enumerate(self.stage_estimators_):
            stage_path = os.path.join(stages_dir, f"stage_{i:04d}.onnx")
            self._export_estimator_to_onnx(tree, self.n_features_, stage_path)
            self.stage_paths_.append(stage_path)
        self.stage_sessions_.clear()

        joblib.dump(
            {
                "meta": {
                    "transform": self.transform,
                    "eps": self.eps,
                    "verbosity": self.verbosity,
                    "n_features": self.n_features_,
                    "random_state": self.random_state,
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "min_samples_leaf": self.min_samples_leaf,
                    "loss_regularization": self.loss_regularization,
                    "subsample": self.subsample,
                    "min_samples_split": self.min_samples_split,
                    "max_features": self.max_features,
                    "max_leaf_nodes": self.max_leaf_nodes,
                    "splitter": self.splitter,
                    "update_tree": self.update_tree,
                    "store_dir": stages_dir,
                    "stage_paths": list(self.stage_paths_),
                },
                "qt": self.qt,
                "pt": self.pt,
                "scaler": self.scaler,
                "_transform_is_fitted": self._transform_is_fitted,
            },
            prefix + "_meta.pkl",
        )

    def load(self, prefix):
        d = joblib.load(prefix + "_meta.pkl")
        meta = d["meta"]
        self.transform = meta["transform"]
        self.eps = meta["eps"]
        self.verbosity = meta["verbosity"]
        self.n_features_ = meta["n_features"]
        self.random_state = meta["random_state"]
        self.n_estimators = meta["n_estimators"]
        self.learning_rate = meta["learning_rate"]
        self.max_depth = meta["max_depth"]
        self.min_samples_leaf = meta["min_samples_leaf"]
        self.loss_regularization = meta["loss_regularization"]
        self.subsample = meta["subsample"]
        self.min_samples_split = meta["min_samples_split"]
        self.max_features = meta["max_features"]
        self.max_leaf_nodes = meta["max_leaf_nodes"]
        self.splitter = meta["splitter"]
        self.update_tree = meta["update_tree"]
        self.store_dir = meta.get("store_dir", None)

        self.qt = d["qt"]
        self.pt = d["pt"]
        self.scaler = d["scaler"]
        self._transform_is_fitted = d["_transform_is_fitted"]

        self.stage_paths_ = list(meta["stage_paths"])
        self.stage_estimators_.clear()
        self.stage_sessions_ = [self._load_onnx_session(p) for p in self.stage_paths_]


# ---------------------------------------
# Iterative reweighter with ONNX export
# ---------------------------------------
class ONNXReweighterMixin:
    """
    Stage-wise multiplicative reweighter:
      F(x) += lr * clip( log((1-p)/p), [-clip, clip] )
      w(x) = w0(x) * exp(F(x))

    Convention: y=1 original, y=0 target.
    """

    def __init__(
        self,
        n_iterations=30,
        mixing_learning_rate=0.1,
        clip_delta=3.0,
        max_log_weight=3.0,
        mixing_subsample=0.7,
        random_state=42,
        store_dir=None,
        reweight_validation_fraction=0.2,
        reweight_early_stopping_rounds=5,
        reweight_metric_every=1,
        # -------- speed knobs (new) --------
        export_onnx_during_fit=False,  # keep False for speed; export only in save()
        update_every=2,  # compute full delta + update F_o every N stages (>=1)
        metric_subsample=0.2,  # fraction of Xo used to compute early-stop metric (0<frac<=1)
        metric_min_points=5000,  # lower bound for metric sample size
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_iterations = int(n_iterations)
        self.mixing_learning_rate = float(mixing_learning_rate)
        self.clip_delta = float(clip_delta)
        self.max_log_weight = float(max_log_weight)
        self.mixing_subsample = float(mixing_subsample)
        self.random_state = int(random_state)
        self.reweight_validation_fraction = float(reweight_validation_fraction)
        self.reweight_early_stopping_rounds = int(reweight_early_stopping_rounds)
        self.reweight_metric_every = max(1, int(reweight_metric_every))

        self.store_dir = (
            store_dir  # optional; save() will override it to a prefix-derived dir
        )
        self.stage_paths_ = []
        self.stage_sessions_ = []

        # in-memory (fast path) — trained estimators per stage
        self.stage_estimators_ = []

        # speed knobs
        self.export_onnx_during_fit = bool(export_onnx_during_fit)
        self.update_every = max(1, int(update_every))
        self.metric_subsample = float(metric_subsample)
        self.metric_min_points = int(metric_min_points)
        self._fit_without_sample_weight_warned = False

    def _use_validation_metric(self):
        return (
            self.reweight_validation_fraction > 0.0
            and self.reweight_early_stopping_rounds > 0
        )

    # --- methods child must implement ---
    def _make_base_estimator(self):
        raise NotImplementedError

    def _export_estimator_to_onnx(self, estimator, n_features, path):
        raise NotImplementedError

    def _estimator_predict_proba(self, estimator, X):
        return estimator.predict_proba(X)[:, 1].astype(np.float32)

    def _get_metric_indices(self, rng, n):
        """
        Choose a subset of indices for metric computation (early stopping proxy).
        """
        if self.metric_subsample >= 0.999:
            return np.arange(n)
        k = int(self.metric_subsample * n)
        k = max(1, min(n, max(k, self.metric_min_points)))
        return rng.choice(n, size=k, replace=False)

    def _split_train_validation(self, X, w, rng):
        n = len(X)
        if n < 10 or not self._use_validation_metric():
            idx = np.arange(n)
            return idx, idx

        n_val = max(1, int(self.reweight_validation_fraction * n))
        if n - n_val < 2:
            idx = np.arange(n)
            return idx, idx

        perm = rng.permutation(n)
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        return train_idx, val_idx

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

    def _validation_metric(self, Xo_val, Xt_val, w_val, tw_val):
        ks_vals = [
            self._weighted_ks_1d(Xo_val[:, j], Xt_val[:, j], w_val, tw_val)
            for j in range(Xo_val.shape[1])
        ]
        return float(np.mean(ks_vals))

    def _estimator_supports_sample_weight(self, estimator):
        try:
            return "sample_weight" in inspect.signature(estimator.fit).parameters
        except (TypeError, ValueError):
            return True

    def _fit_estimator(self, estimator, X, y, sample_weight):
        if self._estimator_supports_sample_weight(estimator):
            try:
                estimator.fit(X, y, sample_weight=sample_weight)
                return
            except TypeError as exc:
                if "sample_weight" not in str(exc):
                    raise
        if self.verbosity >= 1 and not self._fit_without_sample_weight_warned:
            print(
                f"[{self.__class__.__name__}] Base estimator does not support sample_weight "
                "in this sklearn version. Falling back to unweighted stage fits."
            )
            self._fit_without_sample_weight_warned = True
        estimator.fit(X, y)

    # --- core logic ---
    def fit(self, original, target, ow=None, tw=None):
        Xo = self._ensure_2d_float32(original)
        Xt = self._ensure_2d_float32(target)

        ow0 = self._ensure_weights(
            Xo,
            ow,
            allow_negative=not self.require_positive_ow_for_fit,
            context="original training weights",
        )
        tw0 = self._ensure_weights(
            Xt,
            tw,
            allow_negative=not self.require_positive_tw_for_fit,
            context="target training weights",
        )

        # Fit transform ONCE on a mixture (important!)
        X_mix = np.vstack([Xo, Xt])
        self._transform_fit(X_mix)

        Xo = self._transform(Xo)
        Xt = self._transform(Xt)

        rng = np.random.RandomState(self.random_state)

        train_idx_o, val_idx_o = self._split_train_validation(Xo, ow0, rng)
        train_idx_t, val_idx_t = self._split_train_validation(Xt, tw0, rng)

        Xo_train = Xo[train_idx_o]
        Xt_train = Xt[train_idx_t]
        ow_train = ow0[train_idx_o]
        tw_train = tw0[train_idx_t]

        Xo_val = Xo[val_idx_o]
        Xt_val = Xt[val_idx_t]
        ow_val0 = ow0[val_idx_o]
        tw_val = tw0[val_idx_t]

        # running log-correction on ORIGINAL only (target ignored)
        F_o = np.zeros(len(Xo_train), dtype=np.float32)
        pending_F_o = np.zeros_like(F_o, dtype=np.float32)
        F_val = np.zeros(len(Xo_val), dtype=np.float32)

        # clear stages
        self.stage_paths_.clear()
        self.stage_sessions_.clear()
        self.stage_estimators_.clear()

        # helper for subsampling each stage (training set subsample)
        def _subsample_idx(n):
            if self.mixing_subsample >= 0.999:
                return np.arange(n)
            k = max(1, int(self.mixing_subsample * n))
            return rng.choice(n, size=k, replace=False)

        use_validation_metric = self._use_validation_metric()
        patience = max(1, self.reweight_early_stopping_rounds)
        best = np.inf
        bad = 0
        w_dbg = ow_train.copy()

        for stage in range(self.n_iterations):
            idx_o = _subsample_idx(len(Xo_train))
            idx_t = _subsample_idx(len(Xt_train))

            # current original weights = ow0 * exp(F_o), with log-cap
            logw_o = np.log(ow_train + self.eps) + F_o
            logw_o = np.clip(logw_o, -self.max_log_weight, +self.max_log_weight)
            w_o = np.exp(logw_o).astype(np.float32)

            # build training set
            X_stage = np.vstack([Xo_train[idx_o], Xt_train[idx_t]]).astype(np.float32)
            y_stage = np.concatenate(
                [
                    np.ones(len(idx_o), dtype=np.int32),
                    np.zeros(len(idx_t), dtype=np.int32),
                ]
            )

            # weights: original uses current w_o; target uses fixed tw0
            w_stage = np.concatenate([w_o[idx_o], tw_train[idx_t]]).astype(np.float32)
            w_stage = self._normalize_by_class(y_stage, w_stage)

            # Evaluate scale for XGBoost classifier
            w_pos = w_o[idx_o]  # original → y=1
            w_neg = tw_train[idx_t]  # target   → y=0

            sum_pos = float(np.sum(w_pos))
            sum_neg = float(np.sum(w_neg))

            scale_pos_weight = sum_neg / (sum_pos + self.eps)

            # fit base classifier for this stage
            est = self._make_base_estimator()
            if hasattr(est, "set_params") and "scale_pos_weight" in est.get_params():
                est.set_params(scale_pos_weight=scale_pos_weight)

            self._fit_estimator(est, X_stage, y_stage, w_stage)

            # keep estimator (fast path)
            self.stage_estimators_.append(est)

            # ---------- metric on subset ----------
            # ---------- always accumulate stage contribution ----------
            p_o = np.clip(
                self._estimator_predict_proba(est, Xo_train), 1e-6, 1.0 - 1e-6
            )
            delta = (np.log1p(-p_o) - np.log(p_o)).astype(np.float32)
            delta = np.clip(delta, -self.clip_delta, +self.clip_delta)
            pending_F_o += self.mixing_learning_rate * delta

            if use_validation_metric:
                p_val = np.clip(
                    self._estimator_predict_proba(est, Xo_val), 1e-6, 1.0 - 1.0e-6
                )
                delta_val = (np.log1p(-p_val) - np.log(p_val)).astype(np.float32)
                delta_val = np.clip(delta_val, -self.clip_delta, +self.clip_delta)
                F_val += self.mixing_learning_rate * delta_val

            # ---------- flush every N stages ----------
            do_full_update = ((stage + 1) % self.update_every == 0) or (
                stage == self.n_iterations - 1
            )
            if do_full_update:
                F_o += pending_F_o
                pending_F_o.fill(0.0)

            metric = np.nan
            if use_validation_metric and (
                (stage + 1) % self.reweight_metric_every == 0
                or stage == self.n_iterations - 1
            ):
                logw_val = np.log(ow_val0 + self.eps) + F_val
                logw_val = np.clip(logw_val, -self.max_log_weight, +self.max_log_weight)
                w_val = np.exp(logw_val).astype(np.float32)
                metric = self._validation_metric(Xo_val, Xt_val, w_val, tw_val)

                if metric < best * 0.999:
                    best = metric
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        if self.verbosity >= 2:
                            print(
                                f"[ONNXReweighterMixin] Early stopping at stage {stage+1}/{self.n_iterations} "
                                f"(validation mean KS={metric:.5f} did not improve for {bad} checks)"
                            )
                        break

            # NOTE: (1) NO ONNX export/session creation here by default.
            # If you *really* want the old behavior, you can set export_onnx_during_fit=True.
            if self.export_onnx_during_fit:
                if self.store_dir is None:
                    self.store_dir = "iterative_onnx_stages"
                os.makedirs(self.store_dir, exist_ok=True)
                stage_path = os.path.join(self.store_dir, f"stage_{stage:04d}.onnx")
                self._export_estimator_to_onnx(est, self.n_features_, stage_path)
                self.stage_paths_.append(stage_path)
                session = self._load_onnx_session(stage_path)
                self.stage_sessions_.append(session)

            if self.verbosity >= 3 and do_full_update:
                logw_dbg = np.log(ow_train + self.eps) + F_o
                logw_dbg = np.clip(logw_dbg, -self.max_log_weight, +self.max_log_weight)
                w_dbg = np.exp(logw_dbg)
                metric_msg = (
                    f"validation_mean_ks={metric:.3f}"
                    if use_validation_metric and np.isfinite(metric)
                    else "validation_mean_ks=disabled"
                )
                print(
                    f"[ONNXReweighterMixin] stage {stage+1}/{self.n_iterations} "
                    f"{metric_msg} update={'Y' if do_full_update else 'N'} "
                    f"w: mean={w_dbg.mean():.3f} std={w_dbg.std():.3f} "
                    f"min={w_dbg.min():.3f} max={w_dbg.max():.3f}"
                )

        if self.verbosity >= 2:
            print(
                f"[ONNXReweighterMixin] Finished fitting. "
                f"Trained stages={len(self.stage_estimators_)} "
                f"(update_every={self.update_every}, metric_subsample={self.metric_subsample}). "
                f"Final weight stats: mean={w_dbg.mean():.3f} std={w_dbg.std():.3f} "
                f"min={w_dbg.min():.3f} max={w_dbg.max():.3f}"
            )
        return self

    def _ensure_sessions_loaded(self):
        """
        Build onnxruntime sessions from stage_paths_ if needed.
        """
        if self.stage_sessions_:
            return
        if not self.stage_paths_:
            return
        self.stage_sessions_ = [self._load_onnx_session(p) for p in self.stage_paths_]

    def predict_weights(self, X, ow=None):
        X = self._ensure_2d_float32(X)
        ow0 = self._ensure_weights(
            X, ow, allow_negative=True, context="prediction weights"
        )
        X = self._transform(X)

        F = np.zeros(len(X), dtype=np.float32)

        # Prefer ONNX sessions if available (loaded from disk)
        self._ensure_sessions_loaded()
        if self.stage_sessions_:
            for sess in self.stage_sessions_:
                p = self._onnx_get_class1_proba(sess, X)
                delta = (np.log1p(-p) - np.log(p)).astype(np.float32)
                delta = np.clip(delta, -self.clip_delta, +self.clip_delta)
                F += self.mixing_learning_rate * delta

        # Otherwise fallback to in-memory estimators (fast in same process)
        elif self.stage_estimators_:
            for est in self.stage_estimators_:
                p = np.clip(self._estimator_predict_proba(est, X), 1e-6, 1.0 - 1e-6)
                delta = (np.log1p(-p) - np.log(p)).astype(np.float32)
                delta = np.clip(delta, -self.clip_delta, +self.clip_delta)
                F += self.mixing_learning_rate * delta
        else:
            raise RuntimeError(
                "[ONNXReweighterMixin] No stages available. Fit or load the model first."
            )

        logw = np.log(ow0 + self.eps) + F
        logw = np.clip(logw, -self.max_log_weight, +self.max_log_weight)
        return np.exp(logw).astype(np.float32)

    def save(self, prefix):
        """
        Save meta + transformers + export ONNX stages (if not already exported).

        ONNX stage files are written under:  <prefix>_stages/stage_XXXX.onnx
        """
        stages_dir = prefix + "_stages"
        os.makedirs(stages_dir, exist_ok=True)

        # Export ONNX stages if we have estimators (normal fast-training path)
        stage_paths = []
        if self.stage_estimators_:
            for i, est in enumerate(self.stage_estimators_):
                stage_path = os.path.join(stages_dir, f"stage_{i:04d}.onnx")
                self._export_estimator_to_onnx(est, self.n_features_, stage_path)
                stage_paths.append(stage_path)
        else:
            # If estimators are not available, we rely on existing stage_paths_ (e.g. old mode)
            stage_paths = list(self.stage_paths_)

        self.stage_paths_ = stage_paths
        self.stage_sessions_.clear()  # don’t serialize sessions

        meta = {
            "transform": self.transform,
            "eps": self.eps,
            "verbosity": self.verbosity,
            "n_features": self.n_features_,
            "n_iterations": self.n_iterations,
            "mixing_learning_rate": self.mixing_learning_rate,
            "clip_delta": self.clip_delta,
            "max_log_weight": self.max_log_weight,
            "mixing_subsample": self.mixing_subsample,
            "random_state": self.random_state,
            "store_dir": stages_dir,
            "reweight_validation_fraction": self.reweight_validation_fraction,
            "reweight_early_stopping_rounds": self.reweight_early_stopping_rounds,
            "reweight_metric_every": self.reweight_metric_every,
            "stage_paths": self.stage_paths_,
            # speed knobs (persist)
            "export_onnx_during_fit": self.export_onnx_during_fit,
            "update_every": self.update_every,
            "metric_subsample": self.metric_subsample,
            "metric_min_points": self.metric_min_points,
            "weight_norm": self.weight_norm,
        }

        joblib.dump(
            {
                "meta": meta,
                "qt": self.qt,
                "pt": self.pt,
                "scaler": self.scaler,
                "_transform_is_fitted": self._transform_is_fitted,
            },
            prefix + "_meta.pkl",
        )

    def load(self, prefix):
        d = joblib.load(prefix + "_meta.pkl")
        meta = d["meta"]

        self.transform = meta["transform"]
        self.eps = meta["eps"]
        self.verbosity = meta["verbosity"]
        self.n_features_ = meta["n_features"]

        self.n_iterations = meta["n_iterations"]
        self.mixing_learning_rate = meta["mixing_learning_rate"]
        self.clip_delta = meta["clip_delta"]
        self.max_log_weight = meta["max_log_weight"]
        self.mixing_subsample = meta["mixing_subsample"]
        self.random_state = meta["random_state"]
        self.store_dir = meta.get("store_dir", None)
        self.reweight_validation_fraction = float(
            meta.get("reweight_validation_fraction", 0.2)
        )
        self.reweight_early_stopping_rounds = int(
            meta.get("reweight_early_stopping_rounds", 5)
        )
        self.reweight_metric_every = max(1, int(meta.get("reweight_metric_every", 1)))
        self.weight_norm = meta.get("weight_norm", "mean1")

        # speed knobs
        self.export_onnx_during_fit = meta.get("export_onnx_during_fit", False)
        self.update_every = max(1, int(meta.get("update_every", 1)))
        self.metric_subsample = float(meta.get("metric_subsample", 1.0))
        self.metric_min_points = int(meta.get("metric_min_points", 5000))

        self.qt = d["qt"]
        self.pt = d["pt"]
        self.scaler = d["scaler"]
        self._transform_is_fitted = d["_transform_is_fitted"]

        self.stage_paths_ = list(meta["stage_paths"])
        self.stage_estimators_.clear()  # not loaded
        self.stage_sessions_ = [self._load_onnx_session(p) for p in self.stage_paths_]


# ---------------------------------------
# Iterative XGBoost reweighter with ONNX export
# ---------------------------------------
class ONNXIXGBReweighter(ONNXReweighterMixin, BaseONNXReweighter):
    def __init__(
        self,
        transform=None,
        verbosity=1,
        n_iterations=30,
        mixing_learning_rate=0.05,
        clip_delta=2.0,
        max_log_weight=3.0,
        mixing_subsample=1.0,
        random_state=42,
        store_dir=None,
        reweight_validation_fraction=0.2,
        reweight_early_stopping_rounds=5,
        reweight_metric_every=1,
        **xgb_params,
    ):
        self.xgb_params = dict(xgb_params)
        super().__init__(
            transform=transform,
            verbosity=verbosity,
            n_iterations=n_iterations,
            mixing_learning_rate=mixing_learning_rate,
            clip_delta=clip_delta,
            max_log_weight=max_log_weight,
            mixing_subsample=mixing_subsample,
            random_state=random_state,
            store_dir=store_dir,
            reweight_validation_fraction=reweight_validation_fraction,
            reweight_early_stopping_rounds=reweight_early_stopping_rounds,
            reweight_metric_every=reweight_metric_every,
        )
        self.require_positive_ow_for_fit = True
        self.require_positive_tw_for_fit = True
        if self.verbosity >= 1:
            print(
                f"[ONNXIXGBReweighter] Initialized with xgb_params: {self.xgb_params}"
            )

    def _make_base_estimator(self):
        params = dict(
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.8,
            min_child_weight=100.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            base_score=0.5,
            scale_pos_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
        )
        # allow overrides
        params.update(self.xgb_params)
        # remove wrapper keys if they leaked in
        for k in (
            "base_params",
            "base_cls",
            "n_iterations",
            "mixing_learning_rate",
            "mixing_subsample",
            "clip_delta",
            "max_log_weight",
            "store_dir",
            "transform",
            "verbosity",
            "eps",
        ):
            params.pop(k, None)
        return XGBClassifier(**params)

    def _export_estimator_to_onnx(self, estimator, n_features, path):
        update_registered_converter(
            XGBClassifier,
            "XGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        initial = [("input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(
            estimator,
            initial_types=initial,
            target_opset={"ai.onnx.ml": 3, "": 17},
            options={XGBClassifier: {"zipmap": False, "nocl": True}},
        )
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())

    def get_params(self):
        """
        Return the parameters of the reweighter, including both the ONNXReweighterMixin parameters and the XGBoost parameters.
        This can be useful for logging, debugging, or reproducing the model configuration.
        """
        params = dict(
            transform=self.transform,
            eps=self.eps,
            verbosity=self.verbosity,
            n_iterations=self.n_iterations,
            mixing_learning_rate=self.mixing_learning_rate,
            clip_delta=self.clip_delta,
            max_log_weight=self.max_log_weight,
            mixing_subsample=self.mixing_subsample,
            random_state=self.random_state,
            store_dir=self.store_dir,
            reweight_validation_fraction=self.reweight_validation_fraction,
            reweight_early_stopping_rounds=self.reweight_early_stopping_rounds,
            reweight_metric_every=self.reweight_metric_every,
        )
        params.update(self.xgb_params)
        return params

    def set_params(self, **params):
        """
        Set the parameters of the reweighter, allowing updates to both the ONNXReweighterMixin parameters and the XGBoost parameters.
        This can be useful for tuning the model configuration after initialization.

        Args:
            params: Arbitrary keyword arguments corresponding to the parameters to be updated.
        """
        iterkeys = [
            "transform",
            "eps",
            "verbosity",
            "n_iterations",
            "mixing_learning_rate",
            "clip_delta",
            "max_log_weight",
            "mixing_subsample",
            "random_state",
            "store_dir",
            "reweight_validation_fraction",
            "reweight_early_stopping_rounds",
            "reweight_metric_every",
        ]
        for key in iterkeys:
            if key in params:
                setattr(self, key, params[key])
        for key, value in params.items():
            if key not in iterkeys:
                self.xgb_params[key] = value
        if self.verbosity >= 1:
            print(f"[ONNXIXGBReweighter] Updated parameters: {self.get_params()}")


# ---------------------------------------
# Iterative NN reweighter with ONNX export
# ---------------------------------------
class ONNXINNReweighter(ONNXReweighterMixin, BaseONNXReweighter):
    def __init__(
        self,
        transform=None,
        verbosity=1,
        n_iterations=30,
        mixing_learning_rate=0.1,
        clip_delta=3.0,
        max_log_weight=3.0,
        mixing_subsample=1.0,
        random_state=42,
        store_dir=None,
        reweight_validation_fraction=0.2,
        reweight_early_stopping_rounds=5,
        reweight_metric_every=1,
        **nn_params,
    ):
        self.nn_params = dict(nn_params)
        super().__init__(
            transform=transform,
            verbosity=verbosity,
            n_iterations=n_iterations,
            mixing_learning_rate=mixing_learning_rate,
            clip_delta=clip_delta,
            max_log_weight=max_log_weight,
            mixing_subsample=mixing_subsample,
            random_state=random_state,
            store_dir=store_dir,
            reweight_validation_fraction=reweight_validation_fraction,
            reweight_early_stopping_rounds=reweight_early_stopping_rounds,
            reweight_metric_every=reweight_metric_every,
        )
        if self.verbosity >= 1:
            print(f"[ONNXINNReweighter] Initialized with nn_params: {self.nn_params}")

    def _make_base_estimator(self):
        params = dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=150,
            batch_size=1024,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state,
        )
        params.update(self.nn_params)
        # remove wrapper keys if they leaked in
        for k in (
            "base_params",
            "base_cls",
            "n_iterations",
            "mixing_learning_rate",
            "mixing_subsample",
            "clip_delta",
            "max_log_weight",
            "store_dir",
            "transform",
            "verbosity",
            "eps",
        ):
            params.pop(k, None)
        return MLPClassifier(**params)

    def _export_estimator_to_onnx(self, estimator, n_features, path):
        initial = [("input", FloatTensorType([None, n_features]))]
        options = {MLPClassifier: {"zipmap": False, "nocl": True}}
        onx = convert_sklearn(
            estimator, initial_types=initial, target_opset=17, options=options
        )
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())

    def get_params(self):
        """
        Return the parameters of the reweighter, including both the ONNXReweighterMixin parameters and the Neural Network parameters.
        This can be useful for logging, debugging, or reproducing the model configuration.
        """
        params = dict(
            transform=self.transform,
            eps=self.eps,
            verbosity=self.verbosity,
            n_iterations=self.n_iterations,
            mixing_learning_rate=self.mixing_learning_rate,
            clip_delta=self.clip_delta,
            max_log_weight=self.max_log_weight,
            mixing_subsample=self.mixing_subsample,
            random_state=self.random_state,
            store_dir=self.store_dir,
            reweight_validation_fraction=self.reweight_validation_fraction,
            reweight_early_stopping_rounds=self.reweight_early_stopping_rounds,
            reweight_metric_every=self.reweight_metric_every,
        )
        params.update(self.nn_params)
        return params

    def set_params(self, **params):
        """
        Set the parameters of the reweighter, allowing updates to both the ONNXReweighterMixin parameters and the Neural Network parameters.
        This can be useful for tuning the model configuration after initialization.

        Args:
            params: Arbitrary keyword arguments corresponding to the parameters to be updated.
        """
        iterkeys = [
            "transform",
            "eps",
            "verbosity",
            "n_iterations",
            "mixing_learning_rate",
            "clip_delta",
            "max_log_weight",
            "mixing_subsample",
            "random_state",
            "store_dir",
            "reweight_validation_fraction",
            "reweight_early_stopping_rounds",
            "reweight_metric_every",
        ]
        for key in iterkeys:
            if key in params:
                setattr(self, key, params[key])
        for key, value in params.items():
            if key not in iterkeys:
                self.nn_params[key] = value
        if self.verbosity >= 1:
            print(f"[ONNXINNReweighter] Updated parameters: {self.get_params()}")


class ONNXBinsReweighter(BaseONNXReweighter):
    """
    N-dimensional histogram reweighter with simple neighbor smoothing.

    It relies on the common helpers provided by ``BaseONNXReweighter`` for
    feature transformations, input validation, and per-class weight
    normalization.
    """

    def __init__(
        self,
        transform=None,
        verbosity=1,
        n_bins=50,
        n_neighs=2,
        min_in_bin=1.0,
        eps=1e-6,
    ):
        super().__init__(transform=transform, eps=eps, verbosity=verbosity)
        self.n_bins = int(n_bins)
        self.n_neighs = int(n_neighs)
        self.min_in_bin = float(min_in_bin)

        self.edges = None
        self.ratio = None

    # ------------------------------------------------------------
    def _smooth_nd(self, H):
        """
        Simple N-dimensional smoothing by averaging with immediate neighbors.
        Args:
            H (np.ndarray): N-dimensional histogram to be smoothed.
        Returns:
            np.ndarray: Smoothed histogram.
        """
        out = H.copy()
        for _ in range(self.n_neighs):
            tmp = out
            for axis in range(out.ndim):
                tmp = (np.roll(tmp, 1, axis) + tmp + np.roll(tmp, -1, axis)) / 3.0
            out = tmp
        return out

    # ------------------------------------------------------------
    def fit(self, original, target, ow=None, tw=None):
        """
        Fit the reweighter by computing the ratio of N-dimensional histograms of original and target datasets.

        Args:
            original (array-like): Original dataset (e.g., MC samples).
            target (array-like): Target dataset (e.g., real data samples).
            ow (array-like, optional): Sample weights for the original dataset. If None, uniform weights are used.
            tw (array-like, optional): Sample weights for the target dataset. If None, uniform weights are used.
        """
        Xo = self._ensure_2d_float32(original)
        Xt = self._ensure_2d_float32(target)
        ow = self._ensure_weights(Xo, ow)
        tw = self._ensure_weights(Xt, tw)

        # Fit transform ONCE on a representative mixture (important!)
        X_mix = np.vstack([Xo, Xt])
        self._transform_fit(X_mix)

        Xo = self._transform(Xo)
        Xt = self._transform(Xt)

        self.n_features_ = Xo.shape[1]
        if self.n_bins**self.n_features_ > 1e7:
            raise ValueError(
                f"[ERROR] Too many bins: {self.n_bins ** self.n_features_}. "
                f"[ERROR] Reduce n_bins or n_features."
            )

        # Normalize weights similarly to your previous approach:
        # - per-class sum=1, then global mean=1
        # This avoids trivial overall-normalization differences.
        # If you want pure density ratio, remove these two lines.
        y_o = np.ones(len(Xo), dtype=np.int32)
        y_t = np.zeros(len(Xt), dtype=np.int32)
        ow_n = self._normalize_by_class(y_o, ow)
        tw_n = self._normalize_by_class(y_t, tw)
        ow_n /= ow_n.mean()
        tw_n /= tw_n.mean()

        # ---------- quantile binning defined on TARGET (common choice) ----------
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]  # internal quantiles
        self.edges = [np.quantile(Xt[:, d], qs) for d in range(self.n_features_)]

        # ---------- digitize (vectorized) ----------
        # indices in [0, n_bins-1]
        o_idx = np.vstack(
            [np.searchsorted(self.edges[d], Xo[:, d]) for d in range(self.n_features_)]
        ).T
        t_idx = np.vstack(
            [np.searchsorted(self.edges[d], Xt[:, d]) for d in range(self.n_features_)]
        ).T
        o_idx = np.clip(o_idx, 0, self.n_bins - 1)
        t_idx = np.clip(t_idx, 0, self.n_bins - 1)

        shape = (self.n_bins,) * self.n_features_
        H_mc = np.zeros(shape, dtype=np.float32)
        H_data = np.zeros(shape, dtype=np.float32)

        if self.verbosity >= 2:
            print(
                f"[ONNXBinsReweighter] Fitting with {self.n_bins} bins/dim, "
                f"total bins={self.n_bins ** self.n_features_}"
            )
            print(f"[ONNXBinsReweighter] Samples: original={len(Xo)}, target={len(Xt)}")
            print(
                f"[ONNXBinsReweighter] Weights (raw): "
                f"MC mean={ow.mean():.3f} std={ow.std():.3f} | "
                f"Data mean={tw.mean():.3f} std={tw.std():.3f}"
            )

        # ---------- fill histograms with weights ----------
        for idx, w in zip(o_idx, ow_n):
            H_mc[tuple(idx)] += w
        for idx, w in zip(t_idx, tw_n):
            H_data[tuple(idx)] += w

        # ---------- smooth ----------
        H_mc_s = self._smooth_nd(H_mc)
        H_data_s = self._smooth_nd(H_data)

        floor = self.min_in_bin * (1.0 / H_mc_s.size)
        H_mc_safe = np.maximum(H_mc_s, floor)

        # ---------- ratio with regularization ----------
        self.ratio = (H_data_s + self.eps) / (H_mc_safe + self.eps)
        self.ratio = np.clip(self.ratio, 1e-3, 1e2)
        if self.verbosity >= 2:
            print(
                f"[ONNXBinsReweighter] Histograms filled and smoothed. "
                f"MC sum={H_mc_s.sum():.3f} Data sum={H_data_s.sum():.3f} "
                f"Ratio stats: mean={self.ratio.mean():.3f} std={self.ratio.std():.3f} "
                f"min={self.ratio.min():.3e} max={self.ratio.max():.3e}"
            )

    def predict_weights(self, X, ow=None):
        """
        Predict reweighting factors for the input samples by looking up the ratio in the corresponding histogram bin.

        Args:
            X (array-like): Input samples for which to predict weights.
            ow (array-like, optional): Original weights for the input samples. If None, uniform weights are assumed.

        Returns:
            np.ndarray: Predicted reweighting factors for the input samples.
        """
        X = self._ensure_2d_float32(X)
        ow = self._ensure_weights(X, ow)

        X = self._transform(X)

        # digitize
        idx = np.vstack(
            [np.searchsorted(self.edges[d], X[:, d]) for d in range(self.n_features_)]
        ).T
        idx = np.clip(idx, 0, self.n_bins - 1)

        w = np.array([self.ratio[tuple(i)] for i in idx], dtype=np.float32)
        w *= ow  # apply original weights if provided

        return w

    def save(self, prefix):
        """
        Save the histogram edges and ratio to disk, along with meta information.

        Args:
            prefix (str): Prefix for the saved files. The edges and ratio are
                saved to ``<prefix>_edges.npy`` and ``<prefix>_ratio.npy``.
        """
        np.save(prefix + "_edges.npy", self.edges, allow_pickle=True)
        np.save(prefix + "_ratio.npy", self.ratio)

        joblib.dump(
            {
                "meta": {
                    "transform": self.transform,
                    "eps": self.eps,
                    "verbosity": self.verbosity,
                    "n_features": self.n_features_,
                    "n_bins": self.n_bins,
                    "n_neighs": self.n_neighs,
                    "min_in_bin": self.min_in_bin,
                    "weight_norm": self.weight_norm,
                },
                "qt": self.qt,
                "pt": self.pt,
                "scaler": self.scaler,
                "_transform_is_fitted": self._transform_is_fitted,
            },
            prefix + "_meta.pkl",
        )

    def load(self, prefix):
        self.edges = np.load(prefix + "_edges.npy", allow_pickle=True)
        self.ratio = np.load(prefix + "_ratio.npy")

        d = joblib.load(prefix + "_meta.pkl")
        meta = d["meta"]

        self.transform = meta["transform"]
        self.eps = meta["eps"]
        self.verbosity = meta["verbosity"]
        self.n_features_ = meta["n_features"]
        self.n_bins = meta["n_bins"]
        self.n_neighs = meta["n_neighs"]
        self.min_in_bin = meta["min_in_bin"]
        self.weight_norm = meta.get("weight_norm", "mean1")

        self.qt = d.get("qt", None)
        self.pt = d.get("pt", None)
        self.scaler = d.get("scaler", None)
        self._transform_is_fitted = d.get(
            "_transform_is_fitted", self.transform is None
        )
