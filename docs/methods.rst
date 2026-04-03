Reweighting methods
===================

This page describes the reweighting backends implemented in ``mcreweight`` and
how each method computes new MC event weights. Where relevant, differences with
`hep_ml.reweight <https://arogozhnikov.github.io/hep_ml/reweight.html>`_ are noted.

Overview
--------

``mcreweight`` exposes nine user-facing training modes. They fall into four
main families:

``hep_ml``-native methods
  - ``GB``: direct use of ``hep_ml.reweight.GBReweighter``.
  - ``Folding``: direct use of ``hep_ml.reweight.FoldingReweighter`` around
    ``GB``.

ONNX-exportable gradient-boosting methods
  - ``ONNXGB``: custom tree-based reweighter that reproduces the signed-weight
    logic of ``hep_ml`` while remaining exportable to ONNX.
  - ``ONNXFolding``: K-fold ensemble of ``ONNXGB`` models.

Iterative classifier-ratio methods
  - ``XGB``: iterative reweighter that trains an
    ``xgboost.XGBClassifier`` at each stage and converts classifier
    probabilities into multiplicative weight updates.
  - ``XGBFolding``: K-fold ensemble of ``XGB`` models.
  - ``NN``: iterative reweighter that uses a
    ``sklearn.neural_network.MLPClassifier`` at each stage.
  - ``NNFolding``: K-fold ensemble of ``NN`` models.

Histogram method
  - ``Bins``: N-dimensional histogram ratio reweighter with neighbor
    smoothing.

Quick selection guide:

- ``GB`` / ``Folding``: closest to the original ``hep_ml`` package;
- ``ONNXGB`` / ``ONNXFolding``: same boosting logic as ``hep_ml`` but ONNX-exportable;
- ``XGB`` / ``NN``: iterative classifier-ratio correction;
- ``XGBFolding`` / ``NNFolding`` / ``ONNXFolding``: K-fold variants to reduce bias;
- ``Bins``: non-parametric histogram-ratio baseline, best for low dimensions.

All methods follow the same high-level workflow:

1. split MC and data into training and testing subsets;
2. fit the selected reweighter on the training subset;
3. predict new MC weights;
4. clip very large predicted weights to the 99th percentile for stability;
5. save both the trained model and the produced weight arrays.

The training entry points live in ``src/mcreweight/train.py`` and the ONNX-based
implementations live in ``src/mcreweight/models/onnxreweighter.py`` and
``src/mcreweight/models/onnxfolding.py``.

Method-by-method behavior
-------------------------

GB
~~

``GB`` is a thin wrapper around ``hep_ml.reweight.GBReweighter``. All loss and
tree-update logic comes from ``hep_ml``; the trained object is serialized with
``joblib`` and weights are predicted via ``hep_ml``'s own ``predict_weights``.
Use this when compatibility with the original ``hep_ml`` implementation is the
primary requirement.

ONNXGB
~~~~~~

``ONNXGB`` reimplements the ``GBReweighter`` logic with plain scikit-learn
regression trees so that every stage can be exported to ONNX. It is not a
generic classifier-to-ratio method: it mirrors the signed-weight boosting
strategy of ``hep_ml`` directly.

At each stage, MC and data are concatenated, a regression tree is fit on signed
residuals (MC label ``1``, data label ``0``, with per-class weight normalization),
and the leaf values are replaced with the log ratio of target to original
weighted occupancies. The final event weight is ``original_weight * exp(score)``.

The leaf update is regularized as follows:

.. math::

   \Delta_{\mathrm{leaf}} =
   \log\left(w_{\mathrm{target}} + \lambda\right) -
   \log\left(w_{\mathrm{original}} + \lambda\right),

where ``lambda`` is ``loss_regularization``. Adding ``lambda`` prevents infinite
updates in empty or nearly empty leaves and keeps the correction well-behaved.

The key differences from the other methods:

- vs. ``GB``: same intent, different implementation — ``ONNXGB`` uses scikit-learn
  trees instead of the external ``hep_ml`` estimator, enabling ONNX export;
- vs. ``XGB``/``NN``: keeps the signed-weight boosting logic rather than converting
  classifier probabilities into log-ratio updates.

XGB
~~~

``XGB`` estimates the density ratio between data and MC through an iterative
sequence of binary classifiers, rather than reproducing the ``hep_ml`` loss.
A single classifier often captures only the dominant separation; by refitting
after each weight update, the method progressively corrects the residual
mismatch in the already-reweighted sample.

At each iteration :math:`t`:

1. MC events carry their current weights :math:`w_t(x)`;
2. data events keep fixed target weights;
3. an ``xgboost.XGBClassifier`` is trained to distinguish MC from data;
4. its output probability :math:`p_t(x)` for the MC class is converted into a
   log-ratio correction;
5. MC weights are updated multiplicatively.

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

where ``eta`` = ``mixing_learning_rate``, ``c`` = ``clip_delta``, and
``m`` = ``max_log_weight``.

Intuitively, :math:`\delta_t(x)` is positive when the classifier finds the event
more data-like (weight should increase) and negative when it finds it more
MC-like (weight should decrease). The learning rate ``eta`` and clip bounds
prevent any single stage from making an extreme correction.

At each stage ``scale_pos_weight`` is updated to reflect the current weighted
class balance, and negative training weights are clipped to zero for
estimator compatibility.

NN
~~

``NN`` uses exactly the same iterative log-ratio update as ``XGB``, with an
``sklearn.neural_network.MLPClassifier`` as the stage classifier instead of
``XGBClassifier``. All clipping and damping parameters work identically.

If the installed scikit-learn version does not accept ``sample_weight`` in
``MLPClassifier``, the implementation falls back to unweighted stage fits and
prints a warning. Use this method when smooth, non-tree decision boundaries are
preferred.

Bins
~~~~

``Bins`` computes the density ratio as a direct N-dimensional histogram ratio
in transformed feature space:

1. fit the configured feature transform on the combined MC+data sample;
2. define per-variable bin edges from target-data quantiles;
3. fill weighted MC and data histograms;
4. smooth both histograms by averaging with immediate neighbors;
5. compute ``H_data / H_mc`` with epsilon regularization to avoid division by zero;
6. assign each event the ratio value of its bin.

This is the most transparent method in the package. Because bin counts grow
exponentially with the number of dimensions, it is only reliable for a small
number of training variables. In practice it is strongest in one or two
dimensions, can still be useful up to roughly four with enough population, and
should otherwise be treated as a rough baseline rather than the default choice.

Folding variants
~~~~~~~~~~~~~~~~

The ``Folding`` variants (``Folding``, ``ONNXFolding``, ``XGBFolding``,
``NNFolding``) wrap a base reweighter in a K-fold procedure. Each fold is
trained on ``n_folds - 1`` subsets and applied to the held-out subset, so that
every event receives a weight from a model that was not trained on it. This
reduces the bias that arises when weights are predicted on the same data used
for training.

The folding variants differ in how fold predictions are aggregated:

``hep_ml`` folding (``Folding``)
  Delegates to ``hep_ml.reweight.FoldingReweighter``; predictions are
  effectively out-of-fold when the same dataset is passed back in order.

``mcreweight`` ONNX folding (``ONNXFolding``, ``XGBFolding``, ``NNFolding``)
  Trains one model per fold and combines predictions across folds. Available
  aggregation modes:

  - ``weighted_geometric`` (default): geometric mean weighted by the inverse
    of each fold's validation error;
  - ``geometric``: unweighted geometric mean;
  - ``median``: per-event median across folds.

Data visualization and diagnostics
----------------------------------

The training and application pipelines produce a set of standard plots under
``plots/``. These figures are meant to answer slightly different questions:

- are MC and data already mismatched before training;
- does reweighting improve the agreement on the variables used for training;
- does the improvement transfer to variables that were not used for training;
- are the learned weights numerically well behaved;
- can an independent classifier still distinguish reweighted MC from data;
- where in phase space the remaining mismodelling is concentrated;
- which input variables drive the learned correction.

Input and monitoring distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The one-dimensional histogram outputs are the most direct validation plots.

``input_features_training.png`` and ``input_features_testing.png``
  These show the distributions of the training variables before reweighting,
  separately for the train and test splits. They are the baseline mismatch
  plots. Large pull structures here indicate the differences that the
  reweighter is expected to learn.

``input_features_training_transformed.png`` and ``input_features_testing_transformed.png``
  These show the same variables after the optional preprocessing transform
  (for example ``yeo-johnson`` or ``quantile``). They are useful to verify what
  representation the ONNX-capable methods actually see during training.

``other_vars_training.png`` and ``other_vars_testing.png``
  These correspond to the monitoring variables, called ``other_vars`` in the
  output filenames. They are not used to train the reweighter. Instead they are
  held out as a transfer test: if reweighting improves these variables too, the
  correction is more likely to reflect genuine phase-space mismodelling rather
  than simple overfitting of the training inputs.

``input_features_<method>_weighted.png``
  These show the training variables after applying the weights predicted by a
  given method. This is the main post-training check. A good result is one in
  which the reweighted MC moves closer to the data histogram and the pull panel
  becomes more centered around zero.

``other_vars_<method>_weighted.png``
  These show the same post-training comparison for the monitoring variables.
  Improvements here are especially informative because these variables were not
  part of the direct optimization target.

When applying an already trained model, the corresponding output names are
``input_features_reweighted.png`` and ``other_vars_reweighted.png``. They play
the same role, but now for the separately processed output sample.

Correlation matrices
~~~~~~~~~~~~~~~~~~~~

``corr_mc.png`` and ``corr_data.png`` display the pairwise correlation matrices
of the training variables before reweighting.

These plots are useful because one-dimensional agreement is not enough: two
samples can match marginal distributions and still differ strongly in their
joint structure. The correlation matrices give a compact first view of whether
important linear relationships differ between MC and data before training.

Weight distributions
~~~~~~~~~~~~~~~~~~~~

``weight_distributions.png`` shows the distribution of the predicted event
weights for each trained method.

This plot is primarily a stability diagnostic:

- a narrow distribution centered near one usually indicates a mild correction;
- a broad tail can be acceptable, but may signal that the method must strongly
  upweight a small region of phase space;
- extremely long tails or spikes at very large weights are warning signs for
  statistical instability and for downstream analyses that reuse the weights.

Classifier-based diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several plots are built from a fresh classifier trained after reweighting to
separate reweighted MC from data. These are not the reweighters themselves.
They are a common external probe of how distinguishable the two samples remain.

``roc_curve.png``
  This shows the ROC curve of that diagnostic classifier for each method. If
  reweighting is effective, the classifier should struggle to separate the two
  samples, and the curve should move closer to the diagonal. Equivalently, the
  AUC should move closer to 0.5.

``classifier_output.png``
  This shows the classifier-score distributions for reweighted MC and for data.
  It is often easier to interpret than the ROC curve because it directly shows
  whether the diagnostic classifier assigns similar scores to both samples. The
  plot also reports a weighted KS statistic, which summarizes the mismatch
  between the two score distributions.

The term "output distribution" in this context therefore refers to the
distribution of the diagnostic classifier output score, not to the final
physics variables themselves.

2D score and pull maps
~~~~~~~~~~~~~~~~~~~~~~

``score_map_<method>.png``
  This plot shows the mean diagnostic-classifier score in two-dimensional bins
  of all pairs of training variables. It answers the question: in which regions
  of phase space does the diagnostic classifier still find the reweighted MC
  more MC-like or more data-like? Structured hot spots indicate localized
  residual mismodelling even when one-dimensional projections look acceptable.

``pull_map_<method>.png``
  This plot shows the two-dimensional pull,

  .. math::

     \frac{\rho_{\mathrm{data}} - \rho_{\mathrm{MC}}}
          {\sqrt{\sigma^2_{\mathrm{data}} + \sigma^2_{\mathrm{MC}}}},

  in bins of every pair of training variables. A value near zero means local
  agreement within uncertainty, while large positive or negative values point
  to regions where the reweighted MC is still under- or over-populated with
  respect to data.

The difference between the two diagnostics is:

- the score map is classifier-based and tells you where residual separation is
  still easy for a learned discriminator;
- the pull map is histogram-based and tells you where the weighted local event
  densities still disagree.

SHAP feature-importance plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``feature_importance_<method>.png`` shows SHAP summary values for non-folding
methods when ``shap: true`` is enabled.

SHAP stands for SHapley Additive exPlanations. In this context it measures how
much each input variable contributes to the model's predicted log weight for an
event, relative to a reference expectation.

The SHAP beeswarm plot should be read as follows:

- each point is one event;
- the horizontal position is the SHAP value, meaning the signed contribution of
  that feature to increasing or decreasing the predicted log weight;
- the color encodes whether the feature value itself is low or high;
- features higher in the plot have larger overall impact on the model output.

These plots do not by themselves tell you whether a model is "good" or "bad".
They tell you which variables the reweighter is using most strongly to build
its correction and in which direction they influence the learned weights.

Loss function and update mechanics
----------------------------------

Two distinct loss families are used across the methods.

``hep_ml``-style signed boosting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``GB`` and reimplemented by ``ONNXGB``.

The goal is to fit an additive model for :math:`\log(p_{\text{data}}/p_{\text{MC}})`,
the logarithm of the density ratio. At each stage:

- event weights are normalized separately per class;
- the current event importance is updated as ``sample_weight * exp(y * score)``,
  where ``y`` is ``+1`` for MC and ``-1`` for data;
- trees are fit on the absolute normalized weights;
- leaf values are rewritten from the ratio of target to original weighted
  occupancies.

The tree structure captures where the samples differ; the leaf rewrite converts
that structure into a direct density-ratio correction.

Classifier-ratio iterative updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used by ``XGB`` and ``NN``.

Instead of a custom boosting loss, these methods solve a weighted binary
classification problem between MC and data at each stage and convert the
classifier output into a log density-ratio estimate. The sign is intuitive:

- ``p(x) > 0.5`` for the MC class → event looks too MC-like → weight decreases;
- ``p(x) < 0.5`` for the MC class → event looks more data-like → weight increases.

The three numerical controls in the update equations serve distinct purposes:

- ``clip_delta``: prevents any single stage from making an overconfident jump;
- ``max_log_weight``: caps the total accumulated log-weight globally;
- ``mixing_learning_rate``: dampens each stage correction to stabilize training.

Because each stage is trained on the currently reweighted MC, it targets only
the residual mismatch left by previous updates, rather than re-learning the
same dominant discrepancy.

Validation and early stopping
-----------------------------

The iterative ONNX methods (``ONNXGB``, ``XGB``, ``NN``) add a validation loop
that is not part of the original ``hep_ml`` API.

At each stage, the mean weighted Kolmogorov-Smirnov distance across all
training variables is computed on a held-out validation subset. Training stops
early when this mean KS fails to improve for ``reweight_early_stopping_rounds``
consecutive checks.

This provides a physics-motivated stopping criterion: the model stops when it
no longer reduces observable MC-to-data mismatches, not just when classifier
loss plateaus.

Optuna hyperparameter optimization
----------------------------------

When ``n_trials > 0``, ``mcreweight`` runs an Optuna study before the final
training step, supporting ``GB``, ``ONNXGB``, ``XGB``, and ``NN``.

For each trial, the package trains the candidate reweighter, predicts new MC
weights, then measures how well a fresh classifier can still separate the
reweighted MC from data. The objective is the AUC of that diagnostic classifier:
lower is better, since a well-reweighted sample should be harder to distinguish
from data. Studies are run with Optuna's TPE sampler (``seed=42``).

The sampler is Optuna's TPE sampler with ``seed=42``, and the study direction
is ``minimize``.

Cached studies
~~~~~~~~~~~~~~

Optuna studies are cached under ``weightsdir`` as:

.. code-block:: text

   optuna_study_<classifier_type>_<flattened_training_vars>.pkl

If that file already exists, the study is loaded instead of recomputed.

Seed trials
~~~~~~~~~~~

Before optimization starts, one manually chosen initial trial is enqueued:

``GB``
  ``gb_n_estimators=100``, ``gb_learning_rate=0.1``, ``gb_max_depth=5``

``ONNXGB``
  ``gb_n_estimators=100``, ``gb_learning_rate=0.1``, ``gb_max_depth=4``,
  ``min_samples_leaf=200``, ``loss_regularization=5.0``, ``subsample=1.0``

``XGB``
  ``n_iterations=5``, ``mixing_learning_rate=0.1``,
  ``xgb_learning_rate=0.1``, ``max_depth=6``, ``subsample=0.9``,
  ``reg_alpha=1.0``, ``reg_lambda=5.0``

``NN``
  ``n_iterations=5``, ``mixing_learning_rate=0.1``, ``hidden1=64``,
  ``hidden2=32``, ``alpha=1e-4``, ``nn_learning_rate_init=1e-3``,
  ``batch_size=1024``

Search spaces
~~~~~~~~~~~~~

The current Optuna intervals are:

Shared iterative parameters for ``XGB`` and ``NN``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``n_iterations``: integer in ``[5, 25]``
- ``mixing_learning_rate``: log-uniform float in ``[0.05, 0.3]``

``GB`` search space
^^^^^^^^^^^^^^^^^^^

- ``gb_n_estimators``: integer in ``[50, 150]`` with step ``10``
- ``gb_learning_rate``: log-uniform float in ``[0.05, 0.3]``
- ``gb_max_depth``: integer in ``[3, 8]`` with step ``1``

``ONNXGB`` search space
^^^^^^^^^^^^^^^^^^^^^^^

- ``gb_n_estimators``: integer in ``[50, 150]`` with step ``10``
- ``gb_learning_rate``: log-uniform float in ``[0.05, 0.3]``
- ``gb_max_depth``: integer in ``[3, 8]`` with step ``1``
- ``min_samples_leaf``: integer in ``[50, 500]`` with step ``50``
- ``loss_regularization``: log-uniform float in ``[1.0, 20.0]``
- ``subsample``: float in ``[0.5, 1.0]`` with step ``0.1``

``XGB`` base-estimator search space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``xgb_learning_rate``: log-uniform float in ``[0.05, 0.3]``
- ``max_depth``: integer in ``[4, 8]`` with step ``1``
- ``subsample``: float in ``[0.6, 1.0]`` with step ``0.1``
- ``reg_alpha``: float in ``[0.0, 5.0]`` with step ``0.5``
- ``reg_lambda``: float in ``[1.0, 10.0]`` with step ``1``

``NN`` base-estimator search space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``hidden1``: integer in ``[32, 128]`` with step ``16``
- ``hidden2``: integer in ``[16, 64]`` with step ``16``
- ``alpha``: log-uniform float in ``[1e-6, 1e-2]``
- ``nn_learning_rate_init``: log-uniform float in ``[1e-4, 5e-3]``
- ``batch_size``: categorical choice among ``256``, ``512``, ``1024``
- ``max_iter``: integer in ``[50, 180]`` with step ``10``

How tuned parameters are reused
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the study finishes, the final training functions read ``study.best_params``
and map them onto the concrete training backends:

- ``GB`` uses the tuned boosting parameters directly in
  ``hep_ml.reweight.GBReweighter``;
- ``ONNXGB`` runs its own native Optuna objective with
  ``ONNXGBReweighter`` and reuses the tuned tree/update parameters directly in
  the final ONNX-exportable training pass;
- ``XGB`` combines tuned iterative parameters with tuned XGBoost base-estimator
  parameters in ``ONNXIXGBReweighter``;
- ``NN`` combines tuned iterative parameters with tuned MLP base-estimator
  parameters in ``ONNXINNReweighter``.

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
