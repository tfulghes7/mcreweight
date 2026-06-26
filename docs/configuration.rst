Configuration reference
=======================

``mcreweight`` exposes two entry-point commands, ``run-reweight`` and
``apply-weights``.  Both accept a YAML configuration file and support CLI
overrides for every option.  The CLI always takes precedence over the YAML.

.. contents:: On this page
   :local:
   :depth: 2

----

``run-reweight``
----------------

Trains one or more reweighting models and produces diagnostic plots.

Invoke as:

.. code-block:: bash

   run-reweight --config run.yaml [overrides ...]
   run-reweight --dry-run --config run.yaml   # validate config without running

YAML skeleton
~~~~~~~~~~~~~

.. code-block:: yaml

   input:
     mc:
       path: ["/path/to/mc.root"]          # required
       tree: DecayTree                      # default: DecayTree
       mcweights_name: null                 # branch name; null → uniform weights of 1
       mcweights_tree: null                 # separate tree for mcweights_name; null → same tree
       label: MC                            # label used in plots
     data:
       path: ["/path/to/data.root"]        # required
       tree: DecayTree
       sweights_name: sweight_sig           # default: sweight_sig
       sweights_tree: null                  # separate tree for sweights_name; null → same tree
       label: Data
     path_xlabels: null                     # path to YAML of axis labels; null → package defaults

   variables:
     training_vars:                         # required; list of branch names or expressions
       - B_DTF_Jpsi_P
       - B_DTF_Jpsi_PT
       - nPVs
       - nLongTracks
     monitoring_vars: null                  # extra variables to plot but not train on

   reweighting:
     sample: bd_jpsikst_ee                  # subdirectory name under weightsdir and plotdir
     methods:                               # one or more of the values below
       - GB
       - Folding
       - ONNXGB
       - ONNXFolding
       - XGB
       - XGBFolding
       - NN
       - NNFolding
       - Bins
     transform: null                        # quantile | yeo-johnson | signed-log | scaler | null
     n_trials: 10                           # Optuna trials; set to 1 to skip tuning
     test_size: 0.30                        # fraction of events held out for testing
     n_folds: 10                            # number of folds for Folding variants
     n_bins: 10                             # bins per axis for the Bins method
     n_neighs: 3                            # neighbor-smoothing radius for Bins
     reweight_validation_fraction: 0.20     # validation split for iterative early stopping
     reweight_early_stopping_rounds: 5      # patience (consecutive checks without improvement)
     reweight_metric_every: 1               # evaluate validation metric every N stages
     clip_weights: true                     # clip predicted weights at the 99th percentile
     folding_aggregation: weighted_geometric # weighted_geometric | geometric | median
     max_log_weight: 3.0                    # max |log-weight| per event during iterative training
     shap: false                            # compute SHAP feature-importance values

   output:
     weightsdir: null                       # root directory for models and weight arrays;
                                            # falls back to $MCREWEIGHTS_DATA_ROOT if unset
     plotdir: plots                         # root directory for plots

   plotting:
     style: plain                           # plain | LHCb
     sample_label: null                     # text in the top-right of each plot frame (LHCb style only)
     extra_label: null                      # italic text after "LHCb", e.g. Simulation or Preliminary

Key descriptions
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key (YAML path)
     - Description
   * - ``input.mc.path``
     - List of paths to the MC ROOT files.  Multiple files are concatenated.
   * - ``input.mc.tree``
     - Name of the TTree inside each MC file.  Default: ``DecayTree``.
   * - ``input.mc.mcweights_name``
     - Branch name to read per-event MC weights from.  Accepts a plain branch
       name or a mathematical expression built from branch names (e.g.
       ``"w1*w2"``).  When ``null`` all MC events receive weight ``1``.
   * - ``input.mc.mcweights_tree``
     - Name of a separate TTree from which ``mcweights_name`` is read.  Both
       trees must contain the same number of rows.  ``null`` reads from
       ``input.mc.tree``.
   * - ``input.mc.label``
     - Display label used in all plots.  Default: ``MC``.
   * - ``input.data.path``
     - List of paths to the data ROOT files.
   * - ``input.data.tree``
     - Name of the TTree inside each data file.  Default: ``DecayTree``.
   * - ``input.data.sweights_name``
     - Branch name for per-event sWeights (or any data-side weight).  Accepts
       plain names or expressions.  Default: ``sweight_sig``.  Set to ``none``
       to disable sWeights and use uniform data weights instead.
   * - ``input.data.sweights_tree``
     - Separate TTree from which ``sweights_name`` is read.  ``null`` reads
       from ``input.data.tree``.
   * - ``input.data.label``
     - Display label used in all plots.  Default: ``Data``.
   * - ``input.path_xlabels``
     - Path to a YAML file mapping branch names to human-readable axis labels.
       When ``null`` the package's built-in label table is used.
   * - ``variables.training_vars``
     - List of feature names or expressions used to train the reweighter.
       Expressions involving ``+``, ``-``, ``*``, ``/``, ``log``, ``exp`` are
       evaluated with ``numexpr``.
   * - ``variables.monitoring_vars``
     - Additional variables plotted before and after reweighting but not used
       for training.  ``null`` disables monitoring plots.
   * - ``reweighting.sample``
     - Subdirectory name appended to both ``weightsdir`` and ``plotdir`` to
       isolate artifacts per sample.
   * - ``reweighting.methods``
     - Ordered list of reweighting backends to train.  Folding variants require
       the corresponding base method to also be present (e.g. ``Folding``
       requires ``GB``).  Valid values: ``GB``, ``Folding``, ``ONNXGB``,
       ``ONNXFolding``, ``XGB``, ``XGBFolding``, ``NN``, ``NNFolding``,
       ``Bins``.
   * - ``reweighting.transform``
     - Optional feature transform applied before training by all ONNX-capable
       methods.  Choices: ``quantile``, ``yeo-johnson``, ``signed-log``,
       ``scaler``.  ``null`` disables the transform.  The transform is fitted
       once on the combined MC+data training sample and reused at inference.
   * - ``reweighting.n_trials``
     - Number of Optuna trials for hyperparameter search.  Supported for
       ``GB``, ``ONNXGB``, ``XGB``, and ``NN``.  Setting this to ``1``
       skips tuning and uses fixed defaults.
   * - ``reweighting.test_size``
     - Fraction of events reserved for testing (not used during training).
       Default: ``0.3``.
   * - ``reweighting.n_folds``
     - Number of K-folds used by the ``Folding``, ``ONNXFolding``,
       ``XGBFolding``, and ``NNFolding`` methods.  Default: ``10``.
   * - ``reweighting.n_bins``
     - Number of histogram bins per axis for the ``Bins`` method.
       Default: ``10``.
   * - ``reweighting.n_neighs``
     - Neighbor-smoothing radius (in bins) for the ``Bins`` method.
       Default: ``3``.
   * - ``reweighting.reweight_validation_fraction``
     - Fraction of the training set used as a validation sample for early
       stopping in ``ONNXGB``, ``XGB``, and ``NN``.  Default: ``0.2``.
   * - ``reweighting.reweight_early_stopping_rounds``
     - Number of consecutive validation checks without improvement before
       iterative training halts.  Default: ``5``.
   * - ``reweighting.reweight_metric_every``
     - Evaluate the validation KS metric every *N* stages.  Default: ``1``.
   * - ``reweighting.clip_weights``
     - When ``true`` (default), predicted weights are clipped at the 99th
       percentile before saving.  Applies to ``GB``, ``ONNXGB``, ``Folding``,
       ``ONNXFolding``, and ``Bins``.  ``XGB``, ``XGBFolding``, ``NN``, and
       ``NNFolding`` always clip regardless of this flag.
   * - ``reweighting.max_log_weight``
     - Maximum absolute log-weight allowed per event during iterative training
       for ``XGB``, ``XGBFolding``, ``NN``, and ``NNFolding``.  Corresponds to
       a maximum weight ratio of ``exp(max_log_weight)`` (default ``3.0`` →
       ≈ 20×).  Increase this value if the true weight distribution has a heavy
       tail that is being truncated.
   * - ``reweighting.folding_aggregation``
     - How fold-level predictions are combined for ``ONNXFolding``,
       ``XGBFolding``, and ``NNFolding``.  Choices: ``weighted_geometric``
       (default), ``geometric``, ``median``.
   * - ``reweighting.shap``
     - When ``true``, compute SHAP summary values for non-folding methods and
       save feature-importance plots.  Default: ``false``.
   * - ``output.weightsdir``
     - Root directory where trained models and weight arrays are written.  A
       ``<sample>/`` subdirectory is created automatically.  When unset the
       environment variable ``MCREWEIGHTS_DATA_ROOT`` is used as a fallback.
   * - ``output.plotdir``
     - Root directory for diagnostic plots.  A ``<sample>/`` subdirectory is
       created automatically.  Default: ``plots``.
   * - ``plotting.style``
     - Plot style.  ``plain`` (default) uses a clean serif style; ``LHCb``
       applies the mplhep LHCb2 style and adds the experiment label to each
       frame.
   * - ``plotting.sample_label``
     - Text placed in the top-right of each plot frame when ``style`` is
       ``LHCb`` (e.g. a decay-channel label in LaTeX).  Ignored for ``plain``.
   * - ``plotting.extra_label``
     - Italic text rendered immediately after ``LHCb`` on the top-left, e.g.
       ``Simulation`` or ``Preliminary``.  Ignored for ``plain``.

CLI reference
~~~~~~~~~~~~~

All options below override their YAML counterparts when supplied on the command
line.

.. code-block:: text

   run-reweight [--config YAML] [options]

   General
     --config PATH           YAML configuration file
     --dry-run               Validate config and print resolved settings; do not train
     --verbosity {1,2,3,4}   Logging level (default: 1)

   MC input
     --path-mc PATH [PATH …]     Path(s) to MC ROOT file(s)
     --tree-mc TREE              MC TTree name
     --mcweights-name BRANCH     MC weights branch or expression
     --mcweights-tree TREE       Separate tree for MC weights
     --mc-label LABEL            MC label for plots

   Data input
     --path-data PATH [PATH …]   Path(s) to data ROOT file(s)
     --tree-data TREE            Data TTree name
     --sweights-name BRANCH      Data sWeights branch or expression; pass
                                 ``none`` to use uniform data weights
     --sweights-tree TREE        Separate tree for sWeights
     --data-label LABEL          Data label for plots

   Variables
     --training-vars VAR [VAR …]     Training feature names or expressions
     --monitoring-vars VAR [VAR …]   Monitoring variable names (not trained on)

   Reweighting
     --sample NAME                            Sample subdirectory name
     --methods METHOD [METHOD …]              Backends to train; see methods above
     --transform {quantile,yeo-johnson,signed-log,scaler}
                                              Feature transform
     --n_trials INT                           Optuna trials (1 = no tuning)
     --test_size FLOAT                        Test-split fraction
     --n_folds INT                            Number of K-folds
     --n_bins INT                             Bins per axis (Bins method)
     --n_neighs INT                           Neighbor-smoothing radius (Bins method)
     --reweight-validation-fraction FLOAT     Validation fraction for early stopping
     --reweight-early-stopping-rounds INT     Early-stopping patience
     --reweight-metric-every INT              Validate every N stages
     --clip-weights / --clip-weight           Enable weight clipping (flags; default on)
     --max-log-weight FLOAT                   Max |log-weight| per event for XGB/NN (default 3.0)
     --folding-aggregation {weighted_geometric,geometric,median}
                                              Fold-prediction aggregation strategy
     --shap                                   Compute SHAP feature importances

   Output
     --weightsdir DIR     Root directory for model artifacts
     --plotdir DIR        Root directory for plots
     --path-xlabels PATH  YAML file of axis labels

   Plotting
     --style {plain,LHCb}   Plot style
     --sample-label TEXT    Top-right frame label (LHCb style only)
     --extra-label TEXT     Italic text after "LHCb", e.g. Simulation or Preliminary

----

``apply-weights``
-----------------

Applies a previously trained model to a (possibly different) MC sample and
writes the predicted weights back to a ROOT file.

Invoke as:

.. code-block:: bash

   apply-weights --config apply.yaml [overrides ...]
   apply-weights --dry-run --config apply.yaml

YAML skeleton
~~~~~~~~~~~~~

.. code-block:: yaml

   input:
     mc:
       path: ["/path/to/mc_apply.root"]    # required
       tree: DecayTree
       mcweights_name: null
       mcweights_tree: null
       label: MC                            # label used in plots
     data:                                  # optional; enables comparison plots
       path: ["/path/to/data.root"]
       tree: DecayTree
       sweights_name: sweight_sig
       sweights_tree: null
       label: Data                          # label used in plots
     path_xlabels: null

   variables:
     application_vars:                      # variables in the application MC file
       - B_DTF_Jpsi_P
       - B_DTF_Jpsi_PT
       - nPVs
       - nLongTracks
     training_vars:                         # variable names used during training
       - B_DTF_Jpsi_P                       # (must have the same length as application_vars)
       - B_DTF_Jpsi_PT
       - nPVs
       - nLongTracks
     monitoring_vars: null

   reweighting:
     method: XGB                            # single method to apply
     training_sample: bd_jpsikst_ee         # subdirectory where the trained model lives
     application_sample: bd_jpsikst_ee      # subdirectory where output weights are written
     weightsdir: null                       # falls back to $MCREWEIGHTS_DATA_ROOT
     plotdir: plots

   output:
     output_path: "/path/to/output.root"   # required
     output_ntuple: TTree                   # TTree | RNTuple
     output_tree: DecayTree
     weights_name: weights                  # branch name written to the output file

Key descriptions
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key (YAML path)
     - Description
   * - ``input.mc.path``
     - List of paths to the MC ROOT files to apply weights to.
   * - ``input.mc.tree``
     - TTree name.  Default: ``DecayTree``.
   * - ``input.mc.mcweights_name``
     - Prior MC weight branch or expression.  ``null`` → uniform weights of 1.
   * - ``input.mc.mcweights_tree``
     - Separate TTree for ``mcweights_name``.  ``null`` → same as ``input.mc.tree``.
   * - ``input.mc.label``
     - Display label used in comparison plots.  Default: ``MC``.
   * - ``input.data.path``
     - Optional data files.  When provided, comparison distributions are plotted.
   * - ``input.data.tree``
     - Data TTree name.  Default: ``DecayTree``.
   * - ``input.data.sweights_name``
     - Data sWeights branch or expression.  Default: ``sweight_sig``.  Set to
       ``none`` to disable sWeights and use uniform data weights instead.
   * - ``input.data.sweights_tree``
     - Separate TTree for ``sweights_name``.  ``null`` → same as ``input.data.tree``.
   * - ``input.data.label``
     - Display label used in comparison plots.  Default: ``Data``.
   * - ``input.path_xlabels``
     - Path to an axis-label YAML file.  ``null`` → package defaults.
   * - ``variables.application_vars``
     - Branch names (or expressions) in the application MC file to feed into
       the model.  Must have the same length as ``variables.training_vars``; the
       two lists are matched by position to allow renamed branches.
   * - ``variables.training_vars``
     - Feature names used when the model was trained.  These are the column
       names expected by the saved model.
   * - ``variables.monitoring_vars``
     - Extra variables to include in the output plots.
   * - ``reweighting.method``
     - The single backend whose saved model is loaded.  Must match a method
       that was trained in a prior ``run-reweight`` run.  Choices: ``GB``,
       ``Folding``, ``ONNXGB``, ``ONNXFolding``, ``XGB``, ``XGBFolding``,
       ``NN``, ``NNFolding``, ``Bins``.
   * - ``reweighting.training_sample``
     - Subdirectory under ``weightsdir`` where the trained model artifacts live.
       This is typically the ``sample`` value used during training.
   * - ``reweighting.application_sample``
     - Subdirectory under ``weightsdir`` and ``plotdir`` where output weights
       and plots are written.  Can differ from ``training_sample`` when applying
       to a different MC sample.
   * - ``reweighting.weightsdir``
     - Root directory containing trained artifacts.  Falls back to
       ``$MCREWEIGHTS_DATA_ROOT`` when unset.
   * - ``reweighting.plotdir``
     - Root directory for application plots.  Default: ``plots``.
   * - ``output.output_path``
     - Path of the ROOT file to write.  The file is created from scratch; all
       branches from the input MC tree plus the new weights branch are written.
   * - ``output.output_ntuple``
     - Output format: ``TTree`` (default) or ``RNTuple``.
   * - ``output.output_tree``
     - TTree or RNTuple name in the output file.  Default: ``DecayTree``.
   * - ``output.weights_name``
     - Name of the branch that holds the predicted weights in the output file.
       Default: ``weights``.

CLI reference
~~~~~~~~~~~~~

.. code-block:: text

   apply-weights [--config YAML] [options]

   General
     --config PATH           YAML configuration file
     --dry-run               Validate config and print resolved settings; do not run
     --verbosity {1,2,3,4}   Logging level (default: 1)

   MC input
     --path-mc PATH [PATH …]     Path(s) to MC ROOT file(s)
     --tree-mc TREE              MC TTree name
     --mcweights-name BRANCH     MC weights branch or expression
     --mcweights-tree TREE       Separate tree for MC weights
     --mc-label LABEL            MC label for plots

   Data input (optional; enables comparison plots)
     --path-data PATH [PATH …]   Path(s) to data ROOT file(s)
     --tree-data TREE            Data TTree name
     --sweights-name BRANCH      Data sWeights branch or expression; pass
                                 ``none`` to use uniform data weights
     --sweights-tree TREE        Separate tree for sWeights
     --data-label LABEL          Data label for plots

   Variables
     --vars VAR [VAR …]              Application variable names (alias: --vars)
     --training-vars VAR [VAR …]     Training variable names the model expects
     --monitoring-vars VAR [VAR …]   Monitoring variable names

   Reweighting
     --method METHOD                 Backend to apply (see choices above)
     --training-sample NAME          Subdirectory with trained model artifacts
     --application-sample NAME       Subdirectory for output weights and plots
     --weightsdir DIR                Root directory for model artifacts
     --plotdir DIR                   Root directory for plots

   Output
     --output-path PATH      Path for the output ROOT file
     --output-ntuple FORMAT  Output ntuple format: TTree (default) or RNTuple
     --output-tree TREE      Output tree name
     --weights-name BRANCH   Branch name for the predicted weights in the output
     --path-xlabels PATH     YAML file of axis labels

----

Environment variables
---------------------

``MCREWEIGHTS_DATA_ROOT``
  Fallback value for ``weightsdir`` when it is not set in the YAML or on the
  CLI.  Both ``run-reweight`` and ``apply-weights`` raise an error if
  ``weightsdir`` is ultimately unresolvable.

----

Legacy key aliases
------------------

The following YAML keys are recognized for backwards compatibility but are
superseded by the canonical names above:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Legacy key
     - Canonical equivalent
   * - ``input.mc.weights_name``
     - ``input.mc.mcweights_name``
   * - ``input.mc.weights_branch``
     - ``input.mc.mcweights_name``
   * - ``input.data.sweights_branch``
     - ``input.data.sweights_name``
   * - ``input.xlabel_path``
     - ``input.path_xlabels``
   * - ``input.path_xlabel``
     - ``input.path_xlabels``
   * - ``variables.vars``
     - ``variables.application_vars`` (apply-weights only)
   * - ``output.path``
     - ``output.output_path`` (apply-weights only)
   * - ``output.ntuple``
     - ``output.output_ntuple`` (apply-weights only)
   * - ``output.tree``
     - ``output.output_tree`` (apply-weights only)
   * - ``output.weights_branch``
     - ``output.weights_name`` (apply-weights only)
