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
- ``clip_weights: true``
- ``n_trials: 5``
- ``shap: true``

Expected outputs
~~~~~~~~~~~~~~~~

By default the example writes into:

- ``weights/`` for trained models and serialized weight arrays;
- ``plots/`` for validation and diagnostic plots.

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
- write ``plots/training_throughput.json`` summarizing fit timing.

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
must already exist in ``weights/`` before the command can succeed.

The default training example above does not train ``XGB``. To make this example
work, either:

1. run training with a config that includes ``XGB``; or
2. override the application method to one of the methods already trained by
   ``tests_run/run_reweighting_config.yaml``, for example ``ONNXGB``.

Expected outputs
~~~~~~~~~~~~~~~~

For a successful application run, expect:

- a serialized normalized weight array in
  ``weights/mcweights_B_DTF_Jpsi_P_B_DTF_Jpsi_PT_nPVs_nLongTracks.pkl``;
- an output ROOT file named ``test_applied_weights.root``;
- a new branch named ``mult_and_kin_weights_XGB`` in the output tree;
- comparison plots such as:

  ``plots/mc_vars_reweighting.png``
  ``plots/mc_other_vars_reweighting.png``
  ``plots/input_features_reweighted.png``
  ``plots/other_vars_reweighted.png``

The expected behavior is that the output ROOT file keeps the original event
content and adds the requested weight branch for the rows that survived the
input loading mask.

Example 3: throughput sweep
---------------------------

The throughput example uses ``tests_run/throughput_config.yaml`` and is meant to
exercise all available methods on a small sample.

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
- the usual validation plots comparing the different methods.

This is the best example to use when you want to compare backends side by side
or verify that the full method registry is still working.

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
  methods.

In short, the expected qualitative outcome is not a specific number but a set of
artifacts showing that:

- training completed;
- models were saved;
- weights were produced;
- reweighted MC is generally closer to the data than the original MC;
- no method generated obviously pathological weight distributions.
