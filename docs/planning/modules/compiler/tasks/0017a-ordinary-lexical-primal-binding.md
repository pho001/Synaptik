# Task 0017A: Complete ordinary lexical [primal](../../../../glossary.md#term-primal) [binding](../../../../glossary.md#term-binding)

> **Editorial reading key — not part of the historical record.**
>
> The primal resolver is an internal call boundary: FirstOrderAutograd supplies each rule owner a
> function mapping an original operand or saved output to the exact value available in its lexical
> context. Contribution keys still use original identity. Reading a descriptor does not require
> reading a primal; building a value operation does. Supplying fresh detached numerical values in
> a test demonstrates this binding discipline, not permission to replay forward expressions or
> widen Scan eligibility.
>
> Current reading: [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad); [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd).

## Status

Complete — nine reviewed source files integrated against frozen preimages. Independent combined
source/[Javadoc](../../../../glossary.md#term-javadoc) review APPROVE0; MAIN's [Compiler](../../../../glossary.md#term-compiler)/retained-[Scan](../../../../glossary.md#term-cumulative-scan) gate and all unchanged lexical/
ordinary numerical consumers passed. No new Scan [admission](../../../../glossary.md#term-admission) or mathematical rule is claimed.

## Change class

Class B — internal ordinary-rule protocol completion with no new mathematical rule, public [API](../../../../glossary.md#term-application-programming-interface-api),
production type/package, [dependency](../../../../glossary.md#term-dependency), [runtime](../../../../glossary.md#term-runtime) path or [descriptor](../../../../glossary.md#term-descriptor) policy. Clean isolated source,
independent review and MAIN's actual numerical before/after plus affected gate are required.

## Existing authority and problem

[Compiler-owned AD](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [AD glossary](../../../../glossary.md#autograd))</span>
already requires every lexical [primal](../../../../glossary.md#term-primal) operand, including saved outputs, to pass through the
explicit [compile](../../../../glossary.md#compile)-local [binding](../../../../glossary.md#term-binding) function. Existing `FirstOrderAutograd.expandBody(plan,seeds,
primal,constants,normalize)` preserves original-[identity](../../../../glossary.md#term-identity-meanings) contribution keys and [normalization](../../../../glossary.md#term-normalization).
[Elementwise](../../../../glossary.md#term-elementwise)/LinearAlgebra/[Layout](../../../../glossary.md#layout) honor it; Reduction only honors it for SUM. Normalization, [Loss](../../../../glossary.md#loss),
[Convolution](../../../../glossary.md#term-channels-first-convolution-layers), [Pooling](../../../../glossary.md#term-pooling), Indexing, Ordering and Attention still consume original source [Tensors](../../../../glossary.md#tensor).

The unchanged MAIN `LexicalPrimalSmoke` uses real package-private preflightBody/expandBody through
an external reflection-only driver. All original source primals have no storage. Corresponding
actual forward values, including [canonical](../../../../glossary.md#term-canonical-form) auxiliaries, are materialized once into fresh honest
canonical [host](../../../../glossary.md#term-host) values and supplied by the resolver. Nonuniform [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) feed actual ordinary [CPU](../../../../glossary.md#term-cpu)
[evaluation](../../../../glossary.md#term-evaluation) of the resulting [VJP](../../../../glossary.md#term-vector-jacobian-product-vjp); independent [scalar](../../../../glossary.md#term-scalar) [central differences](../../../../glossary.md#term-central-differences) supply expected [gradients](../../../../glossary.md#term-gradient).
Both [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) SUM controls pass. The other 16 cases ([MEAN](../../../../glossary.md#term-mean), Layer/RMS/BatchNorm [inference](../../../../glossary.md#term-inference)/training,
[MSE](../../../../glossary.md#mean-squared-error--mse)/dense CE and joint attention) fail with original-primal missing storage or the SUM-only gate.
The initial unresolved-host-import [fixture](../../../../glossary.md#term-fixture) failure is separately retained, not product evidence.

## Exact source change

Keep ordinary rule algorithms and eligibility/[preflight](../../../../glossary.md#term-preflight) unchanged. Thread the existing
`Function<Tensor,Tensor> primal` as the last [argument](../../../../glossary.md#term-argument) through these package-private [owners](../../../../glossary.md#term-owner):

- Reduction/[Normalization](../../../../glossary.md#term-normalization): `apply(producer,outputIndex,gradient,selectedInputs,constants,primal)`.
- [Loss](../../../../glossary.md#loss)/Indexing: `apply(producer,gradient,selectedInputs,constants,primal)`.
- Attention: `apply(producer,outputIndex,gradient,selectedInputs,constants,primal)`.
- [Convolution](../../../../glossary.md#term-channels-first-convolution-layers): `apply(producer,gradient,selectedInputs,primal)`.
- [Pooling](../../../../glossary.md#term-pooling)/Ordering: `apply(producer,gradient,constants,primal)`.

Migrate the sole production dispatcher and every discovered direct caller. Remove obsolete
unbound/SUM-only overloads and the `primal==IDENTITY` reduction split; retain one resolver-capable
ordinary implementation. Ordinary [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) supplies its existing [identity](../../../../glossary.md#term-identity-meanings) resolver. Keep the
existing row-aware SUM restoration in its [owner](../../../../glossary.md#term-owner), without changing admitted [rank](../../../../glossary.md#term-rank)/[axis](../../../../glossary.md#term-axis) semantics.

Resolve [Tensor](../../../../glossary.md#tensor) operands that feed value operations, not [metadata](../../../../glossary.md#term-metadata) unnecessarily. Preserve original
[producer](../../../../glossary.md#term-producer)/[canonical](../../../../glossary.md#term-canonical-form) index identities, [descriptor](../../../../glossary.md#term-descriptor)-only type/[shape](../../../../glossary.md#shape) reads and original normalization
[targets](../../../../glossary.md#term-target). Resolve integral targets/indices/[masks](../../../../glossary.md#term-boolean-mask) and saved canonical outputs irrespective of their
[gradient](../../../../glossary.md#term-gradient) flags. Specifically include BatchNorm saved [mean](../../../../glossary.md#term-mean)/inverse std slots 3/4, max-pool saved
output, scatter [extrema](../../../../glossary.md#saturation-pole-and-extremum) saved output, TOP_K indices slot1 and attention [weights](../../../../glossary.md#term-weight) slot1. SORT's
matching ARGSORT remains ordinary [derivative](../../../../glossary.md#term-derivative) construction, not replaying SORT/TOP_K. Do not clone
producers, reconstruct forward [activations](../../../../glossary.md#term-activation), introduce a [ValueId](../../../../glossary.md#logical-valueid-versus-physical-memory-slot)-to-Tensor map or copy formulas.

No changes to AutogradPreflight, FirstOrderGradientCoverage, ScanBodyInference/ScanGradientRules
[admission](../../../../glossary.md#term-admission), [Model](../../../../glossary.md#term-model)/header/row facts, [Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal), wire/cold or [runtime](../../../../glossary.md#term-runtime) demand/proof protocols.
Stochastic operations remain outside [Scan](../../../../glossary.md#term-cumulative-scan) purity and are not broadened by this task. Ordinary
unsupported, selected-target, disconnected and higher-order behavior stays exactly as before.

## Files and [ownership](../../../../glossary.md#term-ownership-meanings)

All production files stay in [modules](../../../../glossary.md#term-module)/[compiler](../../../../glossary.md#term-compiler)'s existing package.

- Slice A: FirstOrderAutograd dispatch, ReductionGradientRules, NormalizationGradientRules,
  LossGradientRules and their directly affected existing tests/Javadocs.<span data-glossary-navigation="true"> ([Slice glossary](../../../../glossary.md#slice))</span>
- Slice B: AttentionGradientRules, ConvolutionGradientRules, PoolingGradientRules,
  IndexingGradientRules, OrderingGradientRules and their directly affected existing tests/Javadocs.

One full file has one writer. Two isolated worktrees share an exact accepted dirty-source base;
no concurrent shared file edits. A owns [dispatch](../../../../glossary.md#term-dispatch) calls to B's frozen [signatures](../../../../glossary.md#term-signature) above. No builds/
tests/lint/formatters in either worker; MAIN integrates both reviewed slices before any gate.
MAIN owns shared contracts/plans/[API](../../../../glossary.md#term-application-programming-interface-api)/changelog/evidence. No shared capability-ledger edits.

## Dependencies and integration

- Depends on: completed existing ordinary AD and bounded Scan rule protocol, Compiler0018 and
  current practical RNN baseline; actual before witness; independent bounded plan/Ready review.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn))</span>
- Conflicts with: ordinary gradient-rule/dispatcher writers. Later typed/demand/closure proof and
  row-source consumers wait for this refactor integration. CPU0078/Metal0080 files are disjoint.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Parallel group: [RELEASE](../../../../glossary.md#term-release)-LEXICAL-[AD](../../../../glossary.md#autograd); exact A/B file split and resolver [signatures](../../../../glossary.md#term-signature) above.
- Common base revision: current accepted HEAD plus full dirty-source/authority [manifest](../../../../glossary.md#term-manifest) at [Ready](../../../../glossary.md#term-ready).
- Integration order: reviewed brief/Ready -> complete isolated A/B source -> independent review ->
  SHA-checked MAIN integration -> unchanged actual lexical after/ordinary-regression gate -> docs.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- Integration validation: Compiler tests/Javadocs/jar, affected direct callers, unchanged lexical
  18-case driver and unchanged ordinary F32/F64 scalar/actual-AD driver; existing bounded Scan
  consumer gate as regression. Final repository/Metal qualification remains separate.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Compiler glossary](../../../../glossary.md#term-compiler); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scalar glossary](../../../../glossary.md#term-scalar); [Metal glossary](../../../../glossary.md#term-metal); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

## Acceptance

All 18 unchanged lexical cases pass actual [VJP](../../../../glossary.md#term-vector-jacobian-product-vjp) numbers against the independent [scalar](../../../../glossary.md#term-scalar) [oracle](../../../../glossary.md#term-test-oracle),
including all floating input roles and all three public BatchNorm training roots. Fresh [detached](../../../../glossary.md#detached-model-results)
[primal](../../../../glossary.md#term-primal) values, not a replayed forward source [graph](../../../../glossary.md#graph), must feed the constructed [derivative](../../../../glossary.md#term-derivative) graph.
Retained SUM controls still pass; ordinary [identity](../../../../glossary.md#term-identity-meanings)-resolver requests and existing bounded
[Scan](../../../../glossary.md#term-cumulative-scan)/recurrent workflows do not change their mathematics/ordering/types. Meaningful existing
regressions remain; add permanent tests only for plausible semantic identity/saved-primal defects
using existing numerical conventions, not source [topology](../../../../glossary.md#term-topology)/name/forwarding assertions.

No new Scan capability is claimed from these internal [API](../../../../glossary.md#term-application-programming-interface-api) results. Per-value [cotangent](../../../../glossary.md#cotangent) demand,
pre-ID constructibility/[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) proof, arbitrary-row semantic facts and ordinary [CPU](../../../../glossary.md#term-cpu) helper
composition remain separately mandatory, not erased by resolving operands.

## MAIN [Ready](../../../../glossary.md#term-ready) [admission](../../../../glossary.md#term-admission)

MAIN verified existing ordinary [AD](../../../../glossary.md#autograd) and Compiler0018/practical [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) acceptance, the frozen genuine
18-case before witness with both SUM controls, independent bounded plan APPROVE0, no conflicting
[Compiler](../../../../glossary.md#term-compiler) writer and the exact A/B [signatures](../../../../glossary.md#term-signature)/file split. Source base is HEAD
`07a281edd3086efde2853211b71b48288db027e1` plus the accepted dirty [snapshot](../../../../glossary.md#term-snapshot) in
`compiler0017a-base.json`, shared by two isolated contexts. MAIN alone integrates shared documents
and both reviewed source slices before Compiler/[Javadoc](../../../../glossary.md#term-javadoc)/jar and actual lexical/ordinary [regression](../../../../glossary.md#term-regression)
checks. CPU0078/Metal0080 source is disjoint. No broader typed/row/[Scan](../../../../glossary.md#term-cumulative-scan) capability is admitted.

## Result

Completed the existing resolver protocol in FirstOrderAutograd and the eight ordinary rule
[owners](../../../../glossary.md#term-owner). The obsolete unbound/SUM-only reduction split is removed; original contribution,
[normalization](../../../../glossary.md#term-normalization) and row-role [identities](../../../../glossary.md#term-identity-meanings), formula ordering and ordinary eligibility remain unchanged.
No tests, new production types, public [API](../../../../glossary.md#term-application-programming-interface-api), [backend](../../../../glossary.md#backend) or wire fields were added.

MAIN's source-matched gate completed successfully in 57s: full [Compiler](../../../../glossary.md#term-compiler) tests, Compiler Javadocs/
jar and [CPU](../../../../glossary.md#term-cpu) tests matching CpuGeneralScan/CpuSymbolicScan. Retained XML records:

| Gate | Passed | Skipped |
|---|---:|---:|
| Compiler | 413 | 0 |
| Selected existing CPU [Scan](../../../../glossary.md#term-cumulative-scan) tests | 216 | 20 |

No failures/errors. All 20 CPU skip names are retained in gate evidence; they are not passing
executions. No timing/native-memory/performance conclusion follows.

Unchanged real numerical consumers passed in both CPU types:

| Consumer surface | Cases | Before | After |
|---|---:|---|---|
| Existing body-local [preflight](../../../../glossary.md#term-preflight)/expansion | 18 | SUM controls pass; 16 original-[primal](../../../../glossary.md#term-primal)/SUM-only failures | All pass |
| Ordinary selected-[target](../../../../glossary.md#term-target) lexical plan, including INT/[BOOL](../../../../glossary.md#data-type--datatype)/saved indices | 84 | SUM controls pass; 82 original-primal/SUM-only/masked-attrs failures | All pass |
| [Conv2d](../../../../glossary.md#convolution--conv2d) groups 1/2, [bias](../../../../glossary.md#term-bias)/no bias; MAX/AVGPool2d | 12 | AVG controls pass; 10 original-primal failures | All pass |
| Ordinary public Compiler-backward [regression](../../../../glossary.md#term-regression) | 84 | Previously passed | All pass |

The lexical surfaces use different [planning](../../../../glossary.md#term-planning) inputs; their overlapping [operation](../../../../glossary.md#operation) examples are
not summed into a fabricated unique-operation total. All evaluate actual constructed VJPs with
nonuniform [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) against independent [scalar](../../../../glossary.md#term-scalar) [central differences](../../../../glossary.md#term-central-differences). Original source primals have
no storage; fresh [detached](../../../../glossary.md#detached-model-results) mapped values include [canonical](../../../../glossary.md#term-canonical-form) saved statistics and INT/BOOL controls.
Grouped [convolution](../../../../glossary.md#term-channels-first-convolution-layers) checks every input/[weight](../../../../glossary.md#term-weight)/bias role; max-pool checks saved winner routing.
The external reflection-only drivers are temporary verification, not a production [dispatch](../../../../glossary.md#term-dispatch) API.

The first body-driver [fixture](../../../../glossary.md#term-fixture) incorrectly imported fresh [host](../../../../glossary.md#term-host) arrays with unresolved logical
[layouts](../../../../glossary.md#layout); it is separately recorded as invalid fixture evidence. Corrected canonical [host storage](../../../../glossary.md#host-storage)
is real newly allocated storage, not fake-[contiguous](../../../../glossary.md#term-contiguous) [graph](../../../../glossary.md#graph) [metadata](../../../../glossary.md#term-metadata). Genuine before failures,
source/[oracle](../../../../glossary.md#term-test-oracle) [hashes](../../../../glossary.md#term-hash), after [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), reviewed integration [manifest](../../../../glossary.md#term-manifest) and [JUnit](../../../../glossary.md#term-junit) XML/counts are retained
in coordinator evidence; temporary driver sources are removed after evidence/documentation review.

Preflight/coverage, source typing/rows, new Scan operations, CPU helper composition, selected-demand
and pre-ID closure proof, low arithmetic locking, [Metal](../../../../glossary.md#term-metal), cold/wire and higher-order gates remain
separate. This completed internal prerequisite is not whole-Scan or whole-[release](../../../../glossary.md#term-release) completion.
