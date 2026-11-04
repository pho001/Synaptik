# Task 0047: Ordinary checked [descriptor](../../../../glossary.md#term-descriptor) seam

**Editorial reading key (current context, not a status update).**

This record concerns the historical three-argument `ScanBody.describeOrdinary` seam, not all-PURE
Scan execution or release completion; its aggregate evidence is not a clean full-root pass. The
caller supplies an already canonical operation, ordered input descriptors and output count. A
descriptor describes type, shape, layout and gradient eligibility, not numerical contents.
`ScanBody` selects the ordinary family owner, whose synchronous completion callback returns the same
operation and every canonical output descriptor. “Siblings” are outputs of one operation: a Top-K
check includes indices even when only values are wanted.

Model family guards define meaning; execution providers realize supported routes rather than
redefine it. Semantic ownership does not mean buffer ownership: this seam creates no Tensor identity
or storage, and storage-free expressions neither execute nor own result buffers. Follow
[descriptor-only ordinary
checks](../../../../reference/tensor/general-scan.md#descriptor-only-ordinary-checks) for the
current scoped overload and its limits, and [per-output ordinary coordinate
facts](../../../../reference/tensor/general-scan.md#per-output-ordinary-coordinate-facts) for why
descriptor validity alone proves neither an independent batch-row coordinate nor differentiation
eligibility. Those current contracts do not expand this task’s historical scope.

## Status

Complete — exact reviewed implementation is integrated in MAIN after Model0046 acceptance.
Actual [descriptor](../../../../glossary.md#term-descriptor)-only calls, affected gates and final independent primitive, structured and
[API](../../../../glossary.md#term-application-programming-interface-api)/evidence/documentation reviews passed with zero remaining P1/P2 findings.

## Class, authority and [scope](../../../../glossary.md#term-scope)

Class B: preserve existing ordinary constructor semantics while making their checks usable before
[Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation. Authority remains the [Model owner](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner))</span>
and [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>.
This task grants no new row-[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), differentiability, [backend](../../../../glossary.md#backend), wire or cold-acquisition domain.

The existing checked ordinary selector in `ScanBody` now invokes [descriptor](../../../../glossary.md#term-descriptor)-only family owners
and checks the [canonical](../../../../glossary.md#term-canonical-form) [operation](../../../../glossary.md#operation) and every output descriptor. Real Tensor construction shares
those same [guard](../../../../glossary.md#term-guard)/descriptor cores; there is no copied [shape](../../../../glossary.md#shape)/promotion algorithm, [prototype](../../../../glossary.md#term-prototype) Tensor,
fake identity, [capture](../../../../glossary.md#term-capture-meanings) or alternate [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference).

## Frozen interface

Public seam, in the existing `ScanBody` [owner](../../../../glossary.md#term-owner):

```java
public static List<TensorDescriptor> describeOrdinary(
        Operation operation, List<TensorDescriptor> inputs, int outputCount)
```

The result is [immutable](../../../../glossary.md#term-immutable), complete, [canonical](../../../../glossary.md#term-canonical-form) output order. The input [operation](../../../../glossary.md#operation) is already a
canonical [occurrence](../../../../glossary.md#term-occurrence): reject, rather than normalize away, a different checked operation. Validate
nulls and exact [signature](../../../../glossary.md#term-signature) [cardinality](../../../../glossary.md#term-cardinality) before positional access. Reuse/refactor the existing
ordinary selector instead of retaining a second family switch. Known ordinary [PURE](../../../../glossary.md#term-pure) families only;
custom/unknown kinds, [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../../glossary.md#dropout--inverted-dropout), fixed recurrent scans and nested general scans remain excluded.
[Descriptor](../../../../glossary.md#term-descriptor) [admission](../../../../glossary.md#term-admission) does not establish [Scan](../../../../glossary.md#term-cumulative-scan) [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) or [Compiler](../../../../glossary.md#term-compiler) predicate disposition.

Each existing family exposes this package-private completion entry:

```java
static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
        BiConsumer<Operation, List<TensorDescriptor>> sink)
```

It completes synchronously exactly once using the existing checked-construction [callback](../../../../glossary.md#term-callback) protocol,
with the same canonical operation and all canonical siblings. It creates no [Tensor](../../../../glossary.md#tensor), [producer](../../../../glossary.md#term-producer),
storage or identifier and retains no callback. Preserve the ordinary Tensor entrypoints and their
null/error ordering. Real construction and [metadata](../../../../glossary.md#term-metadata) completion must call the same [guard](../../../../glossary.md#term-guard)/descriptor
cores; do not copy formulas, promotion, [broadcasting](../../../../glossary.md#broadcasting), [normalization](../../../../glossary.md#term-normalization) or geometry. Reuse existing
helpers and attrs constructors. Do not add success-path descriptor lists, wrappers or capturing
closures to real Tensor construction where fixed-[arity](../../../../glossary.md#term-arity) descriptor [parameters](../../../../glossary.md#term-parameter) or existing
collection helpers suffice. Metadata-only result allocation is not an execution-path allocation.

## Integration fields

- Depends on: existing [Model](../../../../glossary.md#term-model) ordinary semantics; Model0046 acceptance before public ScanBody join.
- Conflicts with: writers to the same expression [owner](../../../../glossary.md#term-owner); MAIN owns ScanBody, shared docs and join.
- Parallel group: RELEASE-MODEL47; primitive/layout/index/order owners and reduction/structured
  owners have disjoint exact file lists in the launch receipt.<span data-glossary-navigation="true"> ([RELEASE glossary](../../../../glossary.md#term-release); [layout glossary](../../../../glossary.md#layout))</span>
- Integration order: bounded contract review → exact dirty snapshot → two private-owner slices →
  MAIN single-selector facade → independent source review → one Model/Compiler Javadoc gate and
  actual descriptor-only CLI → final evidence/docs review.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [facade glossary](../../../../glossary.md#term-facade); [Compiler glossary](../../../../glossary.md#term-compiler); [Javadoc glossary](../../../../glossary.md#term-javadoc); [descriptor glossary](../../../../glossary.md#term-descriptor); [CLI glossary](../../../../glossary.md#term-cli))</span>
- Integration validation: existing ordinary Model behavior suite; complete canonical TOP_K,
  five-output batchnorm and attention sibling behavior; ordinary invalid dtype/broadcast/shape
  guards; real descriptor-only calls with no consumed Tensor IDs. No mock/copy/wiring count tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonical glossary](../../../../glossary.md#term-canonical-form); [dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [mock glossary](../../../../glossary.md#term-mock))</span>

## Non-goals and evidence policy

No row-role [heuristic](../../../../glossary.md#term-heuristic), sampled B, [gradient](../../../../glossary.md#term-gradient) math, alternate [derivative](../../../../glossary.md#term-derivative) switch, [Tensor](../../../../glossary.md#tensor) wrapper,
[registry](../../../../glossary.md#term-registry), executor, [CPU](../../../../glossary.md#term-cpu) low-type unlock, numerical [kernel](../../../../glossary.md#kernel) or [persistence](../../../../glossary.md#term-persistence) expansion. No worker
commands/builds/tests/lint/formatters; MAIN owns source-matched verification and exact integration.
Keep source guards separate from independent [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference) and future typed [VJP](../../../../glossary.md#term-vector-jacobian-product-vjp) proof.

## Result

MAIN integrated 88 exact reviewed files: 46 existing family [owners](../../../../glossary.md#term-owner), the single existing ScanBody
selector/[facade](../../../../glossary.md#term-facade), and 41 affected tests. Independent primitive, structured and facade source reviews
report zero remaining findings. Obsolete private-member/count reflection tests and incidental
wording pins were deleted, not repinned; behavioral [shape](../../../../glossary.md#shape)/type/[layout](../../../../glossary.md#layout)/flags/error and pre-ID checks remain.

The original full affected command remains FAILED: 42 [Model](../../../../glossary.md#term-model) failures comprised 41 obsolete
reflection tests and one incidental [overflow](../../../../glossary.md#term-overflow)-message pin. Its [Compiler](../../../../glossary.md#term-compiler) 418, [CPU](../../../../glossary.md#term-cpu) 1,474 passed/
60 existing skips, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 566 and [conformance](../../../../glossary.md#term-conformance) 78 results remain retained. After test migration, the
affected Model-only gate passed 1,169 tests with zero skips. Aggregate affected evidence is 3,705
passes/60 existing CPU skips, not a clean full-root PASS. Production [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), Javadocs and jars passed.

Actual public [descriptor](../../../../glossary.md#term-descriptor) calls covered 84 existing ordinary fixtures and 137 [canonical](../../../../glossary.md#term-canonical-form) occurrences,
all typed casts, independently varied batchnorm/attention flags, Top-K indices, promotion,
broadcast, [view](../../../../glossary.md#term-view-meanings)/materializing reshape, null/[cardinality](../../../../glossary.md#term-cardinality)/shape failures, custom-[PURE](../../../../glossary.md#term-pure) rejection and
[immutable](../../../../glossary.md#term-immutable) outputs. Successful and failing calls consumed no [Tensor](../../../../glossary.md#tensor) IDs. The unchanged CPU [artifact](../../../../glossary.md#term-artifact)
[capture](../../../../glossary.md#term-capture-meanings) additionally preserved 85 files byte-for-byte across 14 qualified families/28 [kernels](../../../../glossary.md#kernel).

Exact source manifests, first-failed XML/[logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), repaired Model gate, actual [CLI](../../../../glossary.md#term-cli) output and CPU
compatibility capture are retained in the production-[run](../../../../glossary.md#term-run) receipts model0047-*. Final independent
primitive, structured and contract/evidence/docs reviews all APPROVE with zero P1/P2. This completes
Model0047 only; no all-PURE [Scan](../../../../glossary.md#term-cumulative-scan) execution or production-[release](../../../../glossary.md#term-release) completion is claimed.

