Reweighting methods
===================

This page summarizes the reweighting backends implemented in ``mcreweight``,
describes how each method updates Monte Carlo event weights, and highlights the
main differences with the reference algorithms documented in
`hep_ml.reweight <https://arogozhnikov.github.io/hep_ml/reweight.html>`_.

Overview
--------

``mcreweight`` exposes nine user-facing training modes:

``GB``
  Direct use of ``hep_ml.reweight.GBReweighter``.

``Folding``
  Direct use of ``hep_ml.reweight.FoldingReweighter`` around ``GB``.

``ONNXGB``
  A custom gradient-boosted tree reweighter that reproduces the signed-weight
  logic of ``hep_ml`` while remaining exportable to ONNX.

``ONNXFolding``
  K-fold ensemble of ``ONNXGB`` models.

``XGB``
  Iterative reweighter that trains an ``xgboost.XGBClassifier`` at each
  stage and converts classifier probabilities into multiplicative weight
  updates.

``XGBFolding``
  K-fold ensemble of ``XGB`` models.

``NN``
  Iterative reweighter that uses a
  ``sklearn.neural_network.MLPClassifier`` at each stage.

``NNFolding``
  K-fold ensemble of ``NN`` models.

``Bins``
  N-dimensional histogram ratio reweighter with neighbor smoothing.

All methods follow the same high-level workflow:

1. split MC and data into training and testing subsets;
2. fit the selected reweighter on the training subset;
3. predict new MC weights;
4. optionally clip very large predicted weights if ``clip_weights`` is enabled;
5. save both the trained model and the produced weight arrays.

The training entry points live in ``src/mcreweight/train.py`` and the ONNX-based
implementations live in ``src/mcreweight/models/onnxreweighter.py`` and
``src/mcreweight/models/onnxfolding.py``.

Method-by-method behavior
-------------------------

GB
~~

``GB`` delegates training to ``hep_ml.reweight.GBReweighter``. This is the
closest option to the canonical ``hep_ml`` implementation. In practice it is the
reference behavior in this package:

- the loss and tree-update rules come from ``hep_ml``;
- the trained object is serialized with ``joblib``;
- weight prediction is done through ``hep_ml``'s own ``predict_weights``.

Choose this method when exact compatibility with the original ``hep_ml``
gradient-boosted reweighter matters more than ONNX exportability.

ONNXGB
~~~~~~

``ONNXGB`` is a custom implementation of the ``GBReweighter`` idea. It is not a
generic classifier-to-ratio method. Instead, it mirrors the signed-weight loss
construction used by ``hep_ml``:

- original and target events are concatenated into one sample;
- labels are ``1`` for original MC and ``0`` for target data;
- signed sample weights are normalized class-by-class;
- each stage trains a regression tree on class/sign residuals;
- the tree leaves are replaced with logarithmic weight-ratio updates;
- the final event weight is ``original_weight * exp(score)``.

The important point is that the tree is first trained to separate the two
classes and is then rewritten at leaf level to encode the local target-to-MC
weight ratio. The regularized leaf update is

.. math::

   \Delta_{\mathrm{leaf}} =
   \log\left(w_{\mathrm{target}} + \lambda\right) -
   \log\left(w_{\mathrm{original}} + \lambda\right),

where ``lambda`` is ``loss_regularization``.

This has two practical effects:

- empty or nearly empty leaves do not produce infinite updates;
- the model stays close to the symmetrized density-ratio logic used in
  ``hep_ml``.

Compared with ``GB``, the main difference is implementation, not intent:

- ``GB`` uses the external ``hep_ml`` estimator directly;
- ``ONNXGB`` reimplements the same idea with plain scikit-learn regression trees
  so that each stage can be exported to ONNX.
- unlike ``XGB``, the fit path keeps the signed-weight logic instead of forcing
  training weights to be positive.

XGB
~~~

``XGB`` uses a different training philosophy. Instead of reproducing the
``hep_ml`` loss, it performs iterative density-ratio correction through a
sequence of binary classifiers.

The iterative design is intentional. A single classifier pass usually captures
only the largest separation between MC and data. After that first correction,
the residual mismodelling changes: events that were obviously mismodelled become
less informative, and smaller localized discrepancies start to dominate. By
refitting after every update, the method keeps chasing the remaining mismatch in
the newly reweighted sample instead of trying to solve the whole density-ratio
problem in one aggressive step.

At iteration :math:`t`:

1. MC events carry their current weights :math:`w_t(x)`;
2. data events keep fixed target weights;
3. a classifier is trained to distinguish MC from data;
4. the classifier output :math:`p_t(x)` for the MC class is converted into a
   log-ratio correction;
5. the MC weights are updated multiplicatively.

The stage update is

.. math::

   \delta_t(x) = \log\frac{1 - p_t(x)}{p_t(x)},

followed by clipping and learning-rate damping:

.. math::

   F_{t+1}(x) = F_t(x) + \eta \cdot
   \mathrm{clip}\left(\delta_t(x), -c, c\right),

and the final weights are

.. math::

   w(x) = \exp\left(\mathrm{clip}(\log w_0(x) + F(x), -m, m)\right).

Here:

- ``eta`` is ``mixing_learning_rate``;
- ``c`` is ``clip_delta``;
- ``m`` is ``max_log_weight``.

In other words, each stage learns a residual correction in log-weight space.
The classifier does not directly predict the final event weight. It predicts
where the current reweighted MC sample is still too dense or too sparse with
respect to data, and that information is converted into the next multiplicative
update.

This method behaves more like iterative classifier-based reweighting than
classical ``hep_ml`` gradient boosting. The base learner is
``xgboost.XGBClassifier``, and the code adjusts ``scale_pos_weight`` at each
stage to reflect the current weighted class balance. For estimator
compatibility, negative training weights are clipped to zero before fitting.

NN
~~

``NN`` follows exactly the same iterative update rule as ``XGB``, but the stage
classifier is a multilayer perceptron:

- base estimator: ``sklearn.neural_network.MLPClassifier``;
- same probability-to-log-ratio update;
- same clipping of stage corrections;
- same cumulative log-weight cap.

Depending on the installed scikit-learn version, ``MLPClassifier`` may not
accept ``sample_weight``. In that case the implementation falls back to
unweighted stage fits and prints a warning.

This backend is useful when smoother, non-tree decision boundaries are desired.
Its statistical objective is still classifier-based density-ratio estimation,
not the specialized ``hep_ml`` boosting loss.

Bins
~~~~

``Bins`` computes a direct histogram ratio in transformed feature space.

The procedure is:

1. fit the configured feature transform on the combined MC+data sample;
2. define per-variable bin edges from target-data quantiles;
3. fill MC and data histograms with weighted counts;
4. smooth both histograms by repeated averaging with immediate neighbors;
5. compute the ratio ``H_data / H_mc`` with a safety floor and epsilon
   regularization;
6. assign each event the ratio of the bin into which it falls.

This is the simplest and most interpretable method in the package, but it
inherits the usual curse of dimensionality of binned reweighting and should only
be trusted in low dimensions.

Folding variants
~~~~~~~~~~~~~~~~

``Folding``, ``ONNXFolding``, ``XGBFolding``, and ``NNFolding`` wrap a base
reweighter inside a K-fold procedure.

The purpose is to reduce bias when predicting weights for the same sample used
to train the reweighter. Each fold is trained on ``n_folds - 1`` subsets and is
validated on the held-out subset.

The folding implementations differ in aggregation:

``hep_ml`` folding
  Uses the behavior of ``hep_ml.reweight.FoldingReweighter``. When the same
  dataset is passed back in the same order, predictions are effectively
  out-of-fold.

``mcreweight`` ONNX folding
  Trains one model per fold, computes a validation metric for each fold, and by
  default combines fold predictions with a weighted geometric mean. Fold weights
  are proportional to the inverse of the fold validation error.

The available aggregation modes for ONNX folding are:

- ``weighted_geometric``;
- ``geometric``;
- ``median``.

Loss function and update mechanics
----------------------------------

There are really two different loss families in this package.

``hep_ml``-style signed boosting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``GB`` and reimplemented by ``ONNXGB``.

The idea is to fit an additive model for the logarithm of the density ratio
between target and original samples. The implementation in ``ONNXGB`` closely
follows the ``hep_ml`` documentation and its signed-weight boosting strategy:

- weights are normalized separately inside each class;
- the current event importance is updated as ``sample_weight * exp(y * score)``;
- tree fitting uses the absolute normalized weights;
- leaf values are recomputed from the ratio of target and original weighted
  occupancies in each leaf.

This makes the tree structure respond to where the samples differ, while the
leaf update converts that structure into a direct correction of the MC density.

Classifier-ratio iterative updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``XGB`` and ``NN``.

These methods do not define a custom tree loss. Instead they repeatedly solve a
weighted binary classification problem between MC and data and convert the
classifier probability into an estimate of the log density ratio. If the
classifier predicts:

- ``p(x) > 0.5`` for the MC class, the event looks too MC-like and its weight is
  pushed down;
- ``p(x) < 0.5`` for the MC class, the event looks more data-like and its weight
  is pushed up.

The clipping controls are important:

- ``clip_delta`` limits any single stage from making an extreme correction;
- ``max_log_weight`` caps the total accumulated log-weight;
- ``mixing_learning_rate`` slows the update to stabilize training.

The iterative procedure is optimizing the reweighting process through several
different kinds of corrections:

- residual density-ratio corrections:
  each stage is trained on the currently reweighted MC sample, so it focuses on
  the mismatch that remains after previous updates rather than re-learning the
  same dominant discrepancy;
- class-imbalance corrections:
  the stage weights are normalized by class, and the XGBoost backend also
  updates ``scale_pos_weight`` dynamically so that the classifier remains well
  behaved as MC weights evolve;
- local correction caps:
  the log-ratio update from each stage is clipped with ``clip_delta`` to avoid
  unstable jumps caused by overconfident classifier outputs;
- global regularization of the accumulated solution:
  ``max_log_weight`` prevents the full sequence of updates from driving event
  weights to numerically extreme values;
- validation-driven correction control:
  the model keeps only making additional corrections while the validation mean
  KS continues to improve.

Validation and early stopping
-----------------------------

The iterative ONNX methods add a validation loop that is not part of the
original ``hep_ml`` API.

During training, a validation subset is held out and the package computes the
mean of the one-dimensional weighted Kolmogorov-Smirnov distances across all
training variables. Early stopping is triggered when this mean KS metric stops
improving for ``reweight_early_stopping_rounds`` checks.

This gives the iterative methods a direct physics-motivated stopping criterion:
the model is not just trying to improve classifier loss, it is trying to reduce
observable mismatches between reweighted MC and target data.

Feature transformations
-----------------------

All custom ONNX-capable methods can apply an optional feature transform before
training:

- ``quantile``;
- ``yeo-johnson``;
- ``signed-log``;
- ``scaler``.

The transform is always fitted once on the combined MC+data sample, then reused
for both training and inference. This is important because it prevents the MC
and data samples from being mapped into different feature spaces.

Main differences with ``hep_ml``
--------------------------------

Relative to the algorithms documented at
`hep_ml.reweight <https://arogozhnikov.github.io/hep_ml/reweight.html>`_, the
main differences in ``mcreweight`` are:

1. ``GB`` and ``Folding`` are direct ``hep_ml`` wrappers, while ``ONNXGB``,
   ``XGB``, ``NN``, and the ONNX folding classes are package-native
   implementations.
2. ``ONNXGB`` aims at behavioral compatibility with ``hep_ml.GBReweighter`` but
   is implemented with exportable stage trees so the trained model can be served
   through ONNX Runtime.
3. ``XGB`` and ``NN`` are not ``hep_ml`` algorithms. They use iterative
   classifier-based log-ratio updates instead of the custom signed boosting loss
   described for ``GBReweighter`` in ``hep_ml``.
4. ``Bins`` is conceptually close to ``hep_ml.BinsReweighter`` but the smoothing
   implementation differs. ``hep_ml`` documents a Gaussian filter, while
   ``mcreweight`` uses repeated averaging with immediate neighbors.
5. The ONNX folding classes use built-in fold scoring and support weighted
   geometric aggregation. The ``hep_ml`` folding interface instead exposes a
   user-provided vote function.
6. The iterative ONNX methods add validation-driven early stopping based on the
   mean weighted KS distance across features. This is not part of the
   ``hep_ml.reweight`` page API.
7. ``mcreweight`` standardizes model persistence across methods and saves
   ONNX-exported stage models for deployment, which is outside the scope of the
   ``hep_ml`` reweighter documentation.

Which method to use
-------------------

As a rule of thumb:

- use ``GB`` if you want the closest behavior to the original ``hep_ml``
  implementation;
- use ``ONNXGB`` if you want similar boosting logic but need ONNX export;
- use ``XGB`` if you want a powerful tree-based iterative classifier reweighter;
- use ``NN`` if a neural iterative classifier is a better inductive bias for the
  problem;
- use folding variants when you will predict on the same sample used for
  training and want less biased event-by-event weights;
- use ``Bins`` only for low-dimensional problems where interpretability matters
  more than flexibility.
