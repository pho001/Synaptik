# Task 0047: Ordinary checked descriptor seam

## Status

Complete — exact reviewed implementation is integrated in MAIN after Model0046 acceptance.
Actual descriptor-only calls, affected gates and final independent primitive, structured and
API/evidence/documentation reviews passed with zero remaining P1/P2 findings.

## Class, authority and scope

Class B: preserve existing ordinary constructor semantics while making their checks usable before
Tensor identity allocation. Authority remains the [Model owner](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
and [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan).
This task grants no new row-coordinate, differentiability, backend, wire or cold-acquisition domain.

The existing checked ordinary selector in `ScanBody` now invokes descriptor-only family owners
and checks the canonical operation and every output descriptor. Real Tensor construction shares
those same guard/descriptor cores; there is no copied shape/promotion algorithm, prototype Tensor,
fake identity, capture or alternate Compiler inference.

## Frozen interface

Public seam, in the existing `ScanBody` owner:

```java
public static List<TensorDescriptor> describeOrdinary(
        Operation operation, List<TensorDescriptor> inputs, int outputCount)
```

The result is immutable, complete, canonical output order. The input operation is already a
canonical occurrence: reject, rather than normalize away, a different checked operation. Validate
nulls and exact signature cardinality before positional access. Reuse/refactor the existing
ordinary selector instead of retaining a second family switch. Known ordinary PURE families only;
custom/unknown kinds, RNG/dropout, fixed recurrent scans and nested general scans remain excluded.
Descriptor admission does not establish Scan coordinates or Compiler predicate disposition.

Each existing family exposes this package-private completion entry:

```java
static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
        BiConsumer<Operation, List<TensorDescriptor>> sink)
```

It completes synchronously exactly once using the existing checked-construction callback protocol,
with the same canonical operation and all canonical siblings. It creates no Tensor, producer,
storage or identifier and retains no callback. Preserve the ordinary Tensor entrypoints and their
null/error ordering. Real construction and metadata completion must call the same guard/descriptor
cores; do not copy formulas, promotion, broadcasting, normalization or geometry. Reuse existing
helpers and attrs constructors. Do not add success-path descriptor lists, wrappers or capturing
closures to real Tensor construction where fixed-arity descriptor parameters or existing
collection helpers suffice. Metadata-only result allocation is not an execution-path allocation.

## Integration fields

- Depends on: existing Model ordinary semantics; Model0046 acceptance before public ScanBody join.
- Conflicts with: writers to the same expression owner; MAIN owns ScanBody, shared docs and join.
- Parallel group: RELEASE-MODEL47; primitive/layout/index/order owners and reduction/structured
  owners have disjoint exact file lists in the launch receipt.
- Integration order: bounded contract review → exact dirty snapshot → two private-owner slices →
  MAIN single-selector facade → independent source review → one Model/Compiler Javadoc gate and
  actual descriptor-only CLI → final evidence/docs review.
- Integration validation: existing ordinary Model behavior suite; complete canonical TOP_K,
  five-output batchnorm and attention sibling behavior; ordinary invalid dtype/broadcast/shape
  guards; real descriptor-only calls with no consumed Tensor IDs. No mock/copy/wiring count tests.

## Non-goals and evidence policy

No row-role heuristic, sampled B, gradient math, alternate derivative switch, Tensor wrapper,
registry, executor, CPU low-type unlock, numerical kernel or persistence expansion. No worker
commands/builds/tests/lint/formatters; MAIN owns source-matched verification and exact integration.
Keep source guards separate from independent Compiler inference and future typed VJP proof.

## Result

MAIN integrated 88 exact reviewed files: 46 existing family owners, the single existing ScanBody
selector/facade, and 41 affected tests. Independent primitive, structured and facade source reviews
report zero remaining findings. Obsolete private-member/count reflection tests and incidental
wording pins were deleted, not repinned; behavioral shape/type/layout/flags/error and pre-ID checks remain.

The original full affected command remains FAILED: 42 Model failures comprised 41 obsolete
reflection tests and one incidental overflow-message pin. Its Compiler 418, CPU 1,474 passed/
60 existing skips, NN 566 and conformance 78 results remain retained. After test migration, the
affected Model-only gate passed 1,169 tests with zero skips. Aggregate affected evidence is 3,705
passes/60 existing CPU skips, not a clean full-root PASS. Production compilation, Javadocs and jars passed.

Actual public descriptor calls covered 84 existing ordinary fixtures and 137 canonical occurrences,
all typed casts, independently varied batchnorm/attention flags, Top-K indices, promotion,
broadcast, view/materializing reshape, null/cardinality/shape failures, custom-PURE rejection and
immutable outputs. Successful and failing calls consumed no Tensor IDs. The unchanged CPU artifact
capture additionally preserved 85 files byte-for-byte across 14 qualified families/28 kernels.

Exact source manifests, first-failed XML/logs, repaired Model gate, actual CLI output and CPU
compatibility capture are retained in the production-run receipts model0047-*. Final independent
primitive, structured and contract/evidence/docs reviews all APPROVE with zero P1/P2. This completes
Model0047 only; no all-PURE Scan execution or production-release completion is claimed.

