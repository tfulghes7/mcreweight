Examples
========

This page shows how to run the example inputs stored in ``tests_run`` and what
outputs to expect from each step.

Fixture files
-------------

The repository ships with small ROOT fixtures:

- ``tests_run/test_mc.root``
- ``tests_run/test_data.root``

These are the inputs used by the example configuration files:

- ``tests_run/run_reweighting_config.yaml``
- ``tests_run/apply_weights_config.yaml``
- ``tests_run/throughput_config.yaml``

If you want to regenerate the fixture ROOT files from larger inputs, use:

.. code-block:: bash

   python tests_run/make_test_root_samples.py \
     --input-data <source_data.root> \
     --input-mc <source_mc.root> \
     --output-data tests_run/test_data.root \
     --output-mc tests_run/test_mc.root \
     --tree DecayTree \
     --n-events 5000

Expected result:

- two ROOT files are written under ``tests_run/``;
- each file contains the first ``n-events`` entries of the requested tree;
- by default the output object type is ``TTree``.

Example 1: train reweighters
----------------------------

The main training example uses ``tests_run/run_reweighting_config.yaml``.

Run it with:

.. code-block:: bash

   run-reweight --config tests_run/run_reweighting_config.yaml

or, in a Pixi environment:

.. code-block:: bash

   pixi run run-reweight --config tests_run/run_reweighting_config.yaml

What this config does
~~~~~~~~~~~~~~~~~~~~~

It trains the following methods on the fixture sample:

- ``ONNXGB``
- ``GB``
- ``NN``
- ``Bins``

using these four training variables:

- ``B_DTF_Jpsi_P``
- ``B_DTF_Jpsi_PT``
- ``nPVs``
- ``nLongTracks``

and these monitoring variables:

- ``B_PHI``
- ``B_ETA``

The config also enables:

- ``transform: yeo-johnson``
- ``n_trials: 5``
- ``shap: true``

Expected outputs
~~~~~~~~~~~~~~~~

By default the example writes into sample-specific subdirectories:

- ``weights/bd_jpsikst_ee/`` for trained models and serialized weight arrays;
- ``plots/bd_jpsikst_ee/`` for validation and diagnostic plots.

.. warning::

   ``Bins`` is included here as a lightweight baseline because the fixture uses
   only four variables. For production use, treat it as a low-dimensional
   method; it is much more fragile than the model-based reweighters once the
   dimensionality or sparsity increases.

For the four configured methods, you should expect model files such as:

- ``weights/gbr_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks.pkl``
- ``weights/onnxgb_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks_meta.pkl``
- ``weights/onnxgb_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks_stages/``
- ``weights/inn_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks_meta.pkl``
- ``weights/inn_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks_stages/``
- ``weights/binning_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks_meta.pkl``
- ``weights/binning_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks_edges.npy``
- ``weights/binning_model_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks_ratio.npy``

and predicted MC weight arrays such as:

- ``weights/gbr_weights_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks.pkl``
- ``weights/onnxgb_weights_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks.pkl``
- ``weights/inn_weights_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks.pkl``
- ``weights/onnx_binning_weights_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks.pkl``

You should also expect diagnostic plots such as:

- ``plots/corr_data.png``
- ``plots/corr_mc.png``
- ``plots/input_features_training.png``
- ``plots/input_features_testing.png``
- ``plots/input_features_training_transformed.png``
- ``plots/input_features_testing_transformed.png``
- ``plots/other_vars_training.png``
- ``plots/other_vars_testing.png``
- ``plots/input_features_gb_weighted.png``
- ``plots/input_features_onnxgb_weighted.png``
- ``plots/input_features_nn_weighted.png``
- ``plots/input_features_binning_weighted.png``
- ``plots/roc_curve.png``
- ``plots/classifier_output.png``
- ``plots/weight_distributions.png``
- ``plots/training_throughput.json``
- ``plots/training_throughput.png``
- ``plots/training_memory.json``
- ``plots/training_memory.png``

In practice, those files are written under ``weights/bd_jpsikst_ee/`` and
``plots/bd_jpsikst_ee/`` because the CLI appends the configured sample name to
the root output directories.

Because ``shap: true`` is enabled, non-folding methods also produce feature
importance plots, for example:

- ``plots/feature_importance_GB.png``
- ``plots/feature_importance_ONNXGB.png``
- ``plots/feature_importance_NN.png``
- ``plots/feature_importance_Bins.png``

What a successful run looks like
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A successful run should:

- read both ROOT inputs without raising I/O errors;
- split the sample into train and test subsets;
- train all requested methods;
- serialize models and weight arrays under ``weights/``;
- create a non-empty set of PNG plots under ``plots/``;
- write ``plots/training_throughput.json`` summarizing fit timing and event
  rates;
- write ``plots/training_memory.json`` summarizing peak resident memory usage
  during each fit.

The exact numerical weights are not fixed, especially for methods with
classifier training or Optuna tuning, but the general expectation is that the
reweighted training and testing distributions should move closer to the target
data sample in the output plots.

Example 2: apply a trained model
--------------------------------

The application example uses ``tests_run/apply_weights_config.yaml``.

Run it with:

.. code-block:: bash

   apply-weights --config tests_run/apply_weights_config.yaml

or:

.. code-block:: bash

   pixi run apply-weights --config tests_run/apply_weights_config.yaml

Important note
~~~~~~~~~~~~~~

This config requests ``method: XGB``. That means the corresponding ``XGB`` model
must already exist in ``weights/bd_jpsikst_ee/`` before the command can succeed.

The default training example above does not train ``XGB``. To make this example
work, either:

1. run training with a config that includes ``XGB``; or
2. override the application method to one of the methods already trained by
   ``tests_run/run_reweighting_config.yaml``, for example ``ONNXGB``.

Expected outputs
~~~~~~~~~~~~~~~~

For a successful application run, expect:

- a serialized normalized weight array in
  ``weights/bd_jpsikst_ee/mcweights_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks.pkl``;
- an output ROOT file named ``test_applied_weights.root``;
- a new branch named ``mult_and_kin_weights_XGB`` in the output tree;
- comparison plots such as:

  ``plots/bd_jpsikst_ee/mc_vars_reweighting.png``
  ``plots/bd_jpsikst_ee/mc_other_vars_reweighting.png``
  ``plots/bd_jpsikst_ee/input_features_reweighted.png``
  ``plots/bd_jpsikst_ee/other_vars_reweighted.png``

The expected behavior is that the output ROOT file keeps the original event
content and adds the requested weight branch for the rows that survived the
input loading mask.

Example 3: throughput and memory sweep
--------------------------------------

The benchmarking example uses ``tests_run/throughput_config.yaml`` and is meant
to exercise all available methods on a small sample while recording both
training speed and memory usage.

Run it with:

.. code-block:: bash

   run-reweight --config tests_run/throughput_config.yaml

This config enables:

- ``GB`` and ``Folding``
- ``ONNXGB`` and ``ONNXFolding``
- ``XGB`` and ``XGBFolding``
- ``NN`` and ``NNFolding``
- ``Bins``

Expected outputs
~~~~~~~~~~~~~~~~

This run should produce:

- one trained model and one weight-array artifact per method;
- ``plots/training_throughput.json`` containing per-method timing and throughput
  summaries;
- ``plots/training_throughput.png`` with a visual summary of relative training
  speed;
- ``plots/training_memory.json`` containing per-method peak RSS summaries;
- ``plots/training_memory.png`` with a visual summary of relative memory
  consumption;
- the usual validation plots comparing the different methods.

This is the best example to use when you want to compare backends side by side
or verify that the full method registry is still working.

What is measured
~~~~~~~~~~~~~~~~

The throughput summary reports:

- fit wall-clock time for each method;
- dataset events per second, defined as the number of training events processed
  per fit second.

The memory summary reports:

- peak RSS (resident set size) reached by the process while fitting each
  method.

Peak RSS is the highest amount of physical memory occupied by the process during
the fit. It is the most useful metric when comparing methods for CI stability
or for estimating whether a given workflow will fit in RAM on a target machine.

Practical ways to reduce runtime and memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a run is too slow or too heavy for the available machine, the most useful
config changes are usually:

- reduce the number of requested ``methods`` and compare backends in separate
  runs instead of training everything at once;
- disable ``shap`` unless feature-importance plots are specifically needed;
- lower ``n_trials`` when using Optuna, since each trial performs an additional
  full training pass;
- avoid folding methods, or lower ``n_folds``, because folding trains multiple
  reweighters per method;
- reduce the number of ``training_vars`` and especially ``monitoring_vars``, as
  all requested columns are loaded into memory and several diagnostics scale
  with the feature count;
- for the ``Bins`` method, reduce ``n_bins`` or the number of input features,
  since the histogram size grows quickly with dimensionality;
- use smaller benchmark-style configs first to compare methods, then rerun only
  the most promising ones on the full sample.

In practice, the easiest low-cost speedup is often to start with a single
method such as ``GB`` or ``XGB``, set ``shap: false``, and keep ``n_trials`` at
``0`` or ``1`` until the rest of the workflow is validated.

Reading the outputs
-------------------

The most useful files to inspect after running the examples are:

- ``plots/input_features_*_weighted.png`` to see whether the reweighted MC moves
  toward the data distribution on the training variables;
- ``plots/other_vars_*_weighted.png`` to see whether improvements transfer to
  monitoring variables not used directly for training;
- ``plots/roc_curve.png`` and ``plots/classifier_output.png`` to assess
  post-reweighting separability;
- ``plots/weight_distributions.png`` to check whether the learned weights are
  numerically well behaved;
- ``plots/training_throughput.json`` to compare computational cost across
  methods;
- ``plots/training_memory.json`` to compare peak memory usage across methods.

In short, the expected qualitative outcome is not a specific number but a set of
artifacts showing that:

- training completed;
- models were saved;
- weights were produced;
- reweighted MC is generally closer to the data than the original MC;
- no method generated obviously pathological weight distributions.
