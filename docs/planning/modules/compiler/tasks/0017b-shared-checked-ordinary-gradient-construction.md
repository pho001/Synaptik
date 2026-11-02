# Task 0017B: Shared checked ordinary gradient construction

## Status

Complete — MAIN integrated seventeen exact source/test paths after independent primitive,
structured and construction reviews, affected gates and actual numerical/lexical/metadata
consumers. Final independent source/evidence/documentation APPROVE0; no new Scan admission follows.

## Goal, class and authority

Class C: factor the eleven existing ordinary PURE VJP owners so one actual formula body can build
real Tensor expressions or prove the same construction using metadata without Tensor identities.
The [Compiler AD owner](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
and [Model guard owner](../../../../architecture/contracts/foundational-modules.md#modulesmodel) remain authoritative.
[ADR0061](../../../../design/decisions/0061-general-scan-ordinary-unit-composition.md) records the broader
composition direction; this task does not widen any Scan source/inference/AD/backend/wire gate.

## Construction contract

- Keep formulas in their existing Elementwise, Reduction, Normalization, LinearAlgebra, Attention,
  Convolution, Pooling, Loss, Layout, Indexing and Ordering owners. Factor each formula once over
  an internal operand type V and a thin ordinary-construction adapter; no shadow derivative switch.
- Real construction uses V=Tensor directly, not a wrapper per Tensor. The adapter delegates to the
  existing ordinary Tensor methods. Preserve operation order, exact coefficients, canonical saved
  slots, selected input positions, cast-back, special-value/tie behavior and constant binding order.
- Preserve the real result-array allocation where practical through a typed array factory, rather
  than generic Object arrays plus copying/wrappers. No runtime/generated-kernel adapter is added.
- Reuse FirstOrderAutograd's existing typed constant owner and coefficient definitions. Metadata
  constants retain exact type/bits and explicit literal roles without creating real constant leaves.
- Metadata construction executes the SAME formula bodies, including eager temporary constructions,
  using ScanBody.describeOrdinary and independently derived CapturedGraphInference descriptors.
  Expose/factor the existing inference dispatch; do not copy it or construct fake nodes/IDs.
  Preserve deferred CropToShape layout compatibility and DISPROVEN/DEFERRED predicate disposition.
- Keep original source-role metadata distinct from lazy resolved primal values. Descriptor access
  alone must not demand a primal. Bind all actual value operands, saved outputs and discrete
  indices/masks through the supplied resolver, retaining original identity contribution keys.
- A synchronous per-created-operation observation seam, for metadata and optional real Tensor
  materialization, may carry future compile-local coordinate/activity facts. It is not another
  semantic graph, registry, persisted certificate, executor, numerical language or Tensor state.
  Cover primal-only generated operations too; ordinary unobserved construction needs no observer allocation.
- Reuse existing argument normalization/guards. If the canonical metadata facade lacks a necessary
  shared raw-argument guard, report that exact prerequisite before copying normalization logic.
  No prototype Tensor, reflection into private constructors, sampled B or compatibility shim.
- Intrinsic trailing axes may be named positively in the SAME formula: attention's three SUM
  sites use rank-minus-1/2/3; pooling's two ARG_MAX sites use the candidate's last axis (rank four).
  Preserve identical canonical operations, without copying general negative-axis normalization.
- Stochastic and joint Scan owners remain on their existing paths. Remove obsolete ordinary-only
  overloads/callers after cutover; do not keep a parallel old formula implementation.

## Files, integration and validation

- Existing production owners under modules/compiler/src/main/java/io/github/pho001/synaptik/compiler:
  FirstOrderAutograd, the eleven ordinary *GradientRules owners, CapturedGraphInference and their
  actual shared geometry helpers. One focused package-private adapter owner is permitted if it
  makes responsibility clearer; no Model public API change without a concrete prerequisite review.
- Depends on: Complete Compiler0017A and Model0046/0047, integrated CPU0078 lock, all verified.
  Exact base: HEAD07a281edd3086efde2853211b71b48288db027e1 plus accepted dirty manifest
  ordinary-gradient-distribution-base.json; MAIN recorded Ready/source ownership before launch.
- Conflicts: those Compiler files and their direct focused tests. MAIN owns shared docs/integration.
  Parallel group RELEASE-ORDINARY-GRADIENT-DISTRIBUTION; build/distribution files are disjoint.
- Order: contract review/Ready -> clean frozen context -> complete factoring -> independent source/
  Javadoc review -> MAIN affected gate and unchanged actual ordinary/lexical VJP consumers -> final
  evidence/docs review and exact integration. Workers run no commands/builds/tests/lint/formatters.
- Preserve existing behavioral tests; delete obsolete reflection/member/count/source-wording pins,
  never repin them. New permanent tests only for plausible consumer-visible boundaries, not forwarding.
- MAIN proves all existing ordinary scalar-oracle/lexical VJP fixtures unchanged and directly runs
  the checked metadata construction, including typed canonical siblings, invalid construction,
  exact constant roles, zero consumed Tensor IDs and lazy primal observations. Retain first failures.
  Old qualified CPU fingerprints/keys/class bytes must remain unchanged for unchanged recipes.

No new derivative mathematics, new admission, low numerical unlock, cold/wire expansion, native
work, runtime helper dispatch or all-PURE completion claim. Report exact changed-file preimage/final
SHAs and interface contract; MAIN owns runtime verification, docs, final review and acceptance.

## Result

Each of the eleven existing ordinary formula owners now has one generic mathematical body.
The real route uses Tensor references directly; one package-private OrdinaryConstruction adapts
ordinary operations and exact existing constants without wrapping each Tensor. The metadata
route checks the same construction through Model's shared guards and the existing independent
Compiler inference dispatcher, including canonical siblings and deferred CropToShape rules.
Typed result arrays, nullable source positions, lazy primal bindings, eager unused attention V
casting, saved BN/attention/TOP_K slots and constant ingress order remain unchanged. Optional
synchronous observation adds no event/inference allocation to the unobserved real branch.

MAIN's first build found seven unmigrated Tensor member chains on generic V; the next found the
missing direct NOT_EQUAL adapter mapping used by pooling. Both failures are retained. The repairs
delegate to the same ordinary operations, not logical-complement or derivative rewrites. Three
independent source/Javadoc slices approved the corrected source with no P1/P2 findings, including
the final documented constructor/source-role delta.

The full affected gate passed: Compiler 423, CPU 1,474 and NN 566 tests, with 60 existing CPU
opt-in skips. Final Compiler Javadocs/JAR passed without the initial three documentation warnings.
The unchanged scalar-oracle public forward/actual-backward corpus passed all 84 cases; the
thirty-case indexing subset is a repeated subset, not additional coverage.

All original lexical numerical consumers passed in strict after mode: 18 targeted, 84 ordinary
and 12 convolution/pooling cases. Earlier cleanup had removed their throwaway source/classes;
initial missing-class launches are retained. MAIN recovered the exact original source writes
and the original resolved-layout/positive-control patch from this session, reproducing the
historical final fixture snapshot tags. No oracle/assertion was relaxed. Initial recovery and
incomplete recompile-classpath failures remain separate from the successful final replay.

A direct checked-construction CLI exercised all eleven ordinary formula families over 100 real
source fixtures: 112 metadata VJP calls produced 184 normalized positional contributions, consuming
zero Tensor identities. Focused permanent behavior tests additionally cover exact literal bits
and binding identity, complete canonical saved-sibling rejection, independent deferred predicates,
Crop layout compatibility and real/metadata eager observation parity.
The unchanged fourteen-family/twenty-eight-kernel capture retained all 85 identity/compatibility/
class/index files byte-for-byte.

Exact source and evidence: `compiler0017b-main-integration.json`,
`compiler0017b-runtime-artifact-evidence.json`, `compiler0017b-metadata-construction-evidence.json`
and retained failed/successful gate logs in the production-run directory. This is ordinary
construction parity, not a finite selected-demand proof, structural cotangent activity plan,
typed Scan AD, physical composition, cold/wire widening, low-compute unlock or complete release.
