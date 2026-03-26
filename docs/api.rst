API reference
=============

This page documents the main public classes and training helpers exposed by
``mcreweight``.

Pipelines
---------

.. autofunction:: mcreweight.core.run_reweighting_pipeline

.. autofunction:: mcreweight.core.apply_weights_pipeline

Training helpers
----------------

.. autofunction:: mcreweight.train.train_and_test

.. autofunction:: mcreweight.train.gbreweight

.. autofunction:: mcreweight.train.onnxgbreweight

.. autofunction:: mcreweight.train.xgbreweight

.. autofunction:: mcreweight.train.nnreweight

.. autofunction:: mcreweight.train.binning_reweight

.. autofunction:: mcreweight.train.gbfolding

.. autofunction:: mcreweight.train.onnxfolding

.. autofunction:: mcreweight.train.xgbfolding

.. autofunction:: mcreweight.train.nnfolding

Main reweighter classes
-----------------------

.. autoclass:: mcreweight.models.onnxreweighter.ONNXGBReweighter
   :members:

.. autoclass:: mcreweight.models.onnxreweighter.ONNXIXGBReweighter
   :members:

.. autoclass:: mcreweight.models.onnxreweighter.ONNXINNReweighter
   :members:

.. autoclass:: mcreweight.models.onnxreweighter.ONNXBinsReweighter
   :members:

Folding classes
---------------

.. autoclass:: mcreweight.models.onnxfolding.ONNXFoldingReweighter
   :members:

.. autoclass:: mcreweight.models.onnxfolding.ONNXIXGBFoldingReweighter
   :members:

.. autoclass:: mcreweight.models.onnxfolding.ONNXINNFoldingReweighter
   :members:

Plotting utilities
------------------

.. autofunction:: mcreweight.utils.plotting_utils.set_lhcb_style

.. autofunction:: mcreweight.utils.plotting_utils.plot_correlation_matrix

.. autofunction:: mcreweight.utils.plotting_utils.plot_distributions

.. autofunction:: mcreweight.utils.plotting_utils.plot_mc_distributions

.. autofunction:: mcreweight.utils.plotting_utils.plot_training_throughput

.. autofunction:: mcreweight.utils.plotting_utils.plot_roc_curve

.. autofunction:: mcreweight.utils.plotting_utils.plot_classifier_output

.. autofunction:: mcreweight.utils.plotting_utils.plot_weight_distributions

.. autofunction:: mcreweight.utils.plotting_utils.plot_2d_score_maps

.. autofunction:: mcreweight.utils.plotting_utils.plot_feature_importance

.. autofunction:: mcreweight.utils.plotting_utils.plot_2d_pull_maps
