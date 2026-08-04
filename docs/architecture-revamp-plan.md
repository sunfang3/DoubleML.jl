# DoubleML architecture revamp

## Decision

Use a hybrid architecture:

- Keep DoubleML responsible for causal data validation, sample splitting,
  cross-fitting semantics, orthogonal scores, influence functions, inference,
  bootstrap, clustering, and sensitivity analysis.
- Use MLJ as an optional learner backend through a package extension.
- Keep the existing built-in learners and public estimator constructors during
  the migration.

MLJ owns ordinary machine-learning concerns. DoubleML must continue to own the
causal concerns. In particular, MLJ resampling must not replace DoubleML's
cluster-aware or repeated sample splitting.

## Why a full package replacement is wrong

The current code is a causal inference engine, not only a learner registry.
Replacing it with GLM.jl, StatsModels.jl, or MLJ would not provide the
orthogonal-score, cross-fitting, influence-function, cluster, or sensitivity
semantics required by the estimators. MLJ is the right boundary for fitting,
prediction, composition, and tuning of nuisance models, but it is not a
replacement for the causal layer.

## Target layers

```text
public estimator API
        |
estimator specification: data roles + score specification
        |
generic DML engine: nuisance tasks + cross-fitting + score evaluation
        |
causal state and inference: IFs, SEs, bootstrap, sensitivity, clustering
        |
learner backend: native learners | MLJ adapter | future backends
```

### Data boundary

Retain `DoubleMLData` as the compatibility boundary. Add validation helpers
and a stable design view, but do not make MLJ's scientific-type conversion part
of the causal data object.

### Cross-fitting boundary

Introduce a typed `CrossFitPlan`/`CrossFitResult` layer for ordinary,
repeated, and cluster-aware folds. The plan owns fold validation and RNG
provenance. A nuisance task describes:

- its name (`:ml_l`, `:ml_m`, `:ml_g0`, ...),
- target values,
- regression or probability prediction,
- optional training-row selector,
- optional fold-specific parameters,
- optional model retention.

This replaces estimator-local copies of the same fit/predict loop. Arm-
specific IRM fits become two nuisance tasks rather than a bespoke loop.

### Score boundary

Introduce `ScoreSpec` with explicit nuisance requirements and a score evaluator
returning `psi_a`, `psi_b`, and optional diagnostics. The score object contains
causal mathematics only; it does not clone learners or know about DataFrames.

PLR and IRM are the first migrations because their score families exercise the
common continuous-outcome, binary-treatment, repeated-fold, clustered,
external-prediction, and sensitivity paths.

### State and inference boundary

Introduce an `EstimationState` container for:

- per-repetition coefficients and standard errors,
- influence functions and score derivatives,
- cross-fitted predictions,
- retained fold models,
- split metadata and diagnostics.

Existing estimator structs continue exposing `coef`, `se`, `psi`, `predictions`,
and related fields while they are migrated. New inference code consumes the
state contract instead of using `hasproperty` to discover capabilities.

### Learner backend boundary

Add an optional `MLJBase` weak dependency and `DoubleMLMLJExt` extension. The
adapter will:

1. clone the configured MLJ model for each fold;
2. build and fit an MLJ machine on the training rows;
3. predict only the held-out rows;
4. normalize regression and probability outputs to DoubleML vectors;
5. retain machines only when requested.

DoubleML remains responsible for the folds. MLJ's own evaluation and
resampling workflows are not used inside DML estimation.

## Migration sequence

1. **Foundation:** add typed cross-fitting, nuisance-task, score, and state
   contracts; route the existing native learner helpers through them.
2. **MLJ bridge:** add the optional extension and tests with MLJBase and
   MLJLinearModels models. Keep the core package loadable without MLJ.
3. **PLR/IRM migration:** convert their fit methods to the generic engine and
   preserve callable scores, external predictions, cluster SEs, sensitivity,
   repeated splits, and stored models.
4. **Linear-score family:** migrate PLIV, IIVM, APO/APOS, and DID variants
   where the nuisance graph is compatible with the generic task model.
5. **Specialized family:** migrate quantile/CVaR, RDD, SSM, panel, and policy
   estimators behind specialized score/task specifications. Do not force these
   into a falsely generic API.
6. **Cleanup:** deprecate duplicated private fit loops and handwritten learner
   parameter plumbing only after compatibility and numerical tests pass.

## Compatibility requirements

- Existing constructors and result fields remain valid through the migration.
- Existing native learners continue working with no MLJ installation.
- External predictions keep their current shape and naming conventions.
- User-provided sample splits remain authoritative.
- Cluster folds remain cluster-aware and are never delegated to generic MLJ
  resampling.
- Random seeds produce reproducible split and fit behavior where the backend
  supports deterministic fitting.
- Numerical tolerance tests compare old and new kernels on fixed splits before
  each estimator family is switched over.

## Acceptance criteria

- Core tests pass with MLJ absent.
- Optional MLJ integration tests pass with MLJBase and MLJLinearModels loaded.
- PLR and IRM produce matching estimates on fixed splits before and after
  migration.
- No estimator fit method owns the ordinary fold-level clone/fit/predict loop
  after migration.
- `hasproperty` checks are removed from the new inference path and reduced to
  compatibility shims only.
- The package load path and precompile time remain acceptable for users who do
  not install MLJ.

## Main risks and mitigations

| Risk | Mitigation |
| --- | --- |
| MLJ probabilistic predictions use distribution objects | Adapter converts by an explicit positive-label contract and tests binary labels. |
| MLJ models mutate machines during fitting | Create one machine per fold and never reuse fitted machines. |
| MLJ resampling conflicts with DML folds | Keep all fold construction in `CrossFitPlan`. |
| Specialized estimators do not share one nuisance graph | Keep specialized task/score specifications instead of over-generalizing. |
| Compatibility fields keep duplication alive | Migrate one family at a time and make state accessors the only new internal path. |

