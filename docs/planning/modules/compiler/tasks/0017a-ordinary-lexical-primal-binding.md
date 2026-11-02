# Task 0017A: Complete ordinary lexical primal binding

## Status

Complete — nine reviewed source files integrated against frozen preimages. Independent combined
source/Javadoc review APPROVE0; MAIN's Compiler/retained-Scan gate and all unchanged lexical/
ordinary numerical consumers passed. No new Scan admission or mathematical rule is claimed.

## Change class

Class B — internal ordinary-rule protocol completion with no new mathematical rule, public API,
production type/package, dependency, runtime path or descriptor policy. Clean isolated source,
independent review and MAIN's actual numerical before/after plus affected gate are required.

## Existing authority and problem

[Compiler-owned AD](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
already requires every lexical primal operand, including saved outputs, to pass through the
explicit compile-local binding function. Existing `FirstOrderAutograd.expandBody(plan,seeds,
primal,constants,normalize)` preserves original-identity contribution keys and normalization.
Elementwise/LinearAlgebra/Layout honor it; Reduction only honors it for SUM. Normalization, Loss,
Convolution, Pooling, Indexing, Ordering and Attention still consume original source Tensors.

The unchanged MAIN `LexicalPrimalSmoke` uses real package-private preflightBody/expandBody through
an external reflection-only driver. All original source primals have no storage. Corresponding
actual forward values, including canonical auxiliaries, are materialized once into fresh honest
canonical host values and supplied by the resolver. Nonuniform seeds feed actual ordinary CPU
evaluation of the resulting VJP; independent scalar central differences supply expected gradients.
Both F32/F64 SUM controls pass. The other 16 cases (MEAN, Layer/RMS/BatchNorm inference/training,
MSE/dense CE and joint attention) fail with original-primal missing storage or the SUM-only gate.
The initial unresolved-host-import fixture failure is separately retained, not product evidence.

## Exact source change

Keep ordinary rule algorithms and eligibility/preflight unchanged. Thread the existing
`Function<Tensor,Tensor> primal` as the last argument through these package-private owners:

- Reduction/Normalization: `apply(producer,outputIndex,gradient,selectedInputs,constants,primal)`.
- Loss/Indexing: `apply(producer,gradient,selectedInputs,constants,primal)`.
- Attention: `apply(producer,outputIndex,gradient,selectedInputs,constants,primal)`.
- Convolution: `apply(producer,gradient,selectedInputs,primal)`.
- Pooling/Ordering: `apply(producer,gradient,constants,primal)`.

Migrate the sole production dispatcher and every discovered direct caller. Remove obsolete
unbound/SUM-only overloads and the `primal==IDENTITY` reduction split; retain one resolver-capable
ordinary implementation. Ordinary compilation supplies its existing identity resolver. Keep the
existing row-aware SUM restoration in its owner, without changing admitted rank/axis semantics.

Resolve Tensor operands that feed value operations, not metadata unnecessarily. Preserve original
producer/canonical index identities, descriptor-only type/shape reads and original normalization
targets. Resolve integral targets/indices/masks and saved canonical outputs irrespective of their
gradient flags. Specifically include BatchNorm saved mean/inverse std slots 3/4, max-pool saved
output, scatter extrema saved output, TOP_K indices slot1 and attention weights slot1. SORT's
matching ARGSORT remains ordinary derivative construction, not replaying SORT/TOP_K. Do not clone
producers, reconstruct forward activations, introduce a ValueId-to-Tensor map or copy formulas.

No changes to AutogradPreflight, FirstOrderGradientCoverage, ScanBodyInference/ScanGradientRules
admission, Model/header/row facts, Prepare/CPU/Metal, wire/cold or runtime demand/proof protocols.
Stochastic operations remain outside Scan purity and are not broadened by this task. Ordinary
unsupported, selected-target, disconnected and higher-order behavior stays exactly as before.

## Files and ownership

All production files stay in modules/compiler's existing package.

- Slice A: FirstOrderAutograd dispatch, ReductionGradientRules, NormalizationGradientRules,
  LossGradientRules and their directly affected existing tests/Javadocs.
- Slice B: AttentionGradientRules, ConvolutionGradientRules, PoolingGradientRules,
  IndexingGradientRules, OrderingGradientRules and their directly affected existing tests/Javadocs.

One full file has one writer. Two isolated worktrees share an exact accepted dirty-source base;
no concurrent shared file edits. A owns dispatch calls to B's frozen signatures above. No builds/
tests/lint/formatters in either worker; MAIN integrates both reviewed slices before any gate.
MAIN owns shared contracts/plans/API/changelog/evidence. No shared capability-ledger edits.

## Dependencies and integration

- Depends on: completed existing ordinary AD and bounded Scan rule protocol, Compiler0018 and
  current practical RNN baseline; actual before witness; independent bounded plan/Ready review.
- Conflicts with: ordinary gradient-rule/dispatcher writers. Later typed/demand/closure proof and
  row-source consumers wait for this refactor integration. CPU0078/Metal0080 files are disjoint.
- Parallel group: RELEASE-LEXICAL-AD; exact A/B file split and resolver signatures above.
- Common base revision: current accepted HEAD plus full dirty-source/authority manifest at Ready.
- Integration order: reviewed brief/Ready -> complete isolated A/B source -> independent review ->
  SHA-checked MAIN integration -> unchanged actual lexical after/ordinary-regression gate -> docs.
- Integration validation: Compiler tests/Javadocs/jar, affected direct callers, unchanged lexical
  18-case driver and unchanged ordinary F32/F64 scalar/actual-AD driver; existing bounded Scan
  consumer gate as regression. Final repository/Metal qualification remains separate.
- Shared-document integration owner: MAIN.

## Acceptance

All 18 unchanged lexical cases pass actual VJP numbers against the independent scalar oracle,
including all floating input roles and all three public BatchNorm training roots. Fresh detached
primal values, not a replayed forward source graph, must feed the constructed derivative graph.
Retained SUM controls still pass; ordinary identity-resolver requests and existing bounded
Scan/recurrent workflows do not change their mathematics/ordering/types. Meaningful existing
regressions remain; add permanent tests only for plausible semantic identity/saved-primal defects
using existing numerical conventions, not source topology/name/forwarding assertions.

No new Scan capability is claimed from these internal API results. Per-value cotangent demand,
pre-ID constructibility/coordinate proof, arbitrary-row semantic facts and ordinary CPU helper
composition remain separately mandatory, not erased by resolving operands.

## MAIN Ready admission

MAIN verified existing ordinary AD and Compiler0018/practical RNN acceptance, the frozen genuine
18-case before witness with both SUM controls, independent bounded plan APPROVE0, no conflicting
Compiler writer and the exact A/B signatures/file split. Source base is HEAD
`07a281edd3086efde2853211b71b48288db027e1` plus the accepted dirty snapshot in
`compiler0017a-base.json`, shared by two isolated contexts. MAIN alone integrates shared documents
and both reviewed source slices before Compiler/Javadoc/jar and actual lexical/ordinary regression
checks. CPU0078/Metal0080 source is disjoint. No broader typed/row/Scan capability is admitted.

## Result

Completed the existing resolver protocol in FirstOrderAutograd and the eight ordinary rule
owners. The obsolete unbound/SUM-only reduction split is removed; original contribution,
normalization and row-role identities, formula ordering and ordinary eligibility remain unchanged.
No tests, new production types, public API, backend or wire fields were added.

MAIN's source-matched gate completed successfully in 57s: full Compiler tests, Compiler Javadocs/
jar and CPU tests matching CpuGeneralScan/CpuSymbolicScan. Retained XML records:

| Gate | Passed | Skipped |
|---|---:|---:|
| Compiler | 413 | 0 |
| Selected existing CPU Scan tests | 216 | 20 |

No failures/errors. All 20 CPU skip names are retained in gate evidence; they are not passing
executions. No timing/native-memory/performance conclusion follows.

Unchanged real numerical consumers passed in both CPU types:

| Consumer surface | Cases | Before | After |
|---|---:|---|---|
| Existing body-local preflight/expansion | 18 | SUM controls pass; 16 original-primal/SUM-only failures | All pass |
| Ordinary selected-target lexical plan, including INT/BOOL/saved indices | 84 | SUM controls pass; 82 original-primal/SUM-only/masked-attrs failures | All pass |
| Conv2d groups 1/2, bias/no bias; MAX/AVGPool2d | 12 | AVG controls pass; 10 original-primal failures | All pass |
| Ordinary public Compiler-backward regression | 84 | Previously passed | All pass |

The lexical surfaces use different planning inputs; their overlapping operation examples are
not summed into a fabricated unique-operation total. All evaluate actual constructed VJPs with
nonuniform seeds against independent scalar central differences. Original source primals have
no storage; fresh detached mapped values include canonical saved statistics and INT/BOOL controls.
Grouped convolution checks every input/weight/bias role; max-pool checks saved winner routing.
The external reflection-only drivers are temporary verification, not a production dispatch API.

The first body-driver fixture incorrectly imported fresh host arrays with unresolved logical
layouts; it is separately recorded as invalid fixture evidence. Corrected canonical host storage
is real newly allocated storage, not fake-contiguous graph metadata. Genuine before failures,
source/oracle hashes, after logs, reviewed integration manifest and JUnit XML/counts are retained
in coordinator evidence; temporary driver sources are removed after evidence/documentation review.

Preflight/coverage, source typing/rows, new Scan operations, CPU helper composition, selected-demand
and pre-ID closure proof, low arithmetic locking, Metal, cold/wire and higher-order gates remain
separate. This completed internal prerequisite is not whole-Scan or whole-release completion.
