Reweighting methods
===================

This page summarizes the reweighting backends implemented in ``mcreweight``,
describes how each method updates Monte Carlo event weights, and highlights the
main differences with the reference algorithms documented in
`hep_ml.reweight <https://arogozhnikov.github.io/hep_ml/reweight.html>`_.

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

Another useful way to compare the methods is by implementation strategy:

- choose ``GB`` or ``Folding`` if you want the closest behavior to the original
  ``hep_ml`` package;
- choose ``ONNXGB`` or ``ONNXFolding`` if you want hep_ml-like boosting logic
  together with ONNX export;
- choose ``XGB`` or ``NN`` if you want iterative density-ratio correction based
  on a sequence of classifiers;
- choose ``XGBFolding``, ``NNFolding``, or ``ONNXFolding`` if you want the same
  base logic but with K-fold training to reduce application bias;
- choose ``Bins`` if you want a transparent non-parametric baseline based on
  binned ratios rather than learned trees or neural networks.

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

``input_features_training_transformed.png`` and
``input_features_testing_transformed.png``
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

Optuna hyperparameter optimization
----------------------------------

When ``n_trials`` is greater than zero, ``mcreweight`` runs an Optuna study
before the final training step. The tuning logic is implemented in
``src/mcreweight/optuna.py`` and supports four classifier families:

- ``GB``
- ``ONNXGB``
- ``XGB``
- ``NN``

The study objective is the output of
``mcreweight.utils.utils.evaluate_reweighting`` applied after reweighting. In
practice, for each trial the package:

1. samples a set of hyperparameters;
2. trains the candidate reweighter on the full MC and data samples;
3. predicts reweighted MC event weights;
4. evaluates how well a fresh classifier can still separate reweighted MC from
   data;
5. minimizes the resulting score.

Because ``evaluate_reweighting`` returns the AUC of a classifier trained to
distinguish reweighted MC from data, smaller values are better. A perfectly
matched reweighted sample should be harder to distinguish from data than a
poorly reweighted one.

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
