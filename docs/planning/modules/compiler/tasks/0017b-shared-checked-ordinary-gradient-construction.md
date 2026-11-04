# Task 0017B: Shared checked ordinary [gradient](../../../../glossary.md#term-gradient) construction

## Status

Complete — MAIN integrated seventeen exact source/test paths after independent primitive,
structured and construction reviews, affected gates and actual numerical/lexical/[metadata](../../../../glossary.md#term-metadata)
consumers. Final independent source/evidence/documentation APPROVE0; no new [Scan](../../../../glossary.md#term-cumulative-scan) [admission](../../../../glossary.md#term-admission) follows.

## Goal, class and authority

Class C: factor the eleven existing ordinary [PURE](../../../../glossary.md#term-pure) [VJP](../../../../glossary.md#term-vector-jacobian-product-vjp) [owners](../../../../glossary.md#term-owner) so one actual formula body can build
real [Tensor](../../../../glossary.md#tensor) expressions or prove the same construction using [metadata](../../../../glossary.md#term-metadata) without Tensor [identities](../../../../glossary.md#term-identity-meanings).
The [Compiler AD owner](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [owner glossary](../../../../glossary.md#term-owner))</span>
and [Model guard owner](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [guard glossary](../../../../glossary.md#term-guard))</span> remain authoritative.
[ADR0061](../../../../design/decisions/0061-general-scan-ordinary-unit-composition.md) records the broader
composition direction; this task does not widen any [Scan](../../../../glossary.md#term-cumulative-scan) source/[inference](../../../../glossary.md#term-inference)/[AD](../../../../glossary.md#autograd)/[backend](../../../../glossary.md#backend)/wire gate.

## Construction contract

- Keep formulas in their existing Elementwise, Reduction, Normalization, LinearAlgebra, Attention,
  Convolution, Pooling, Loss, Layout, Indexing and Ordering owners. Factor each formula once over
  an internal operand type V and a thin ordinary-construction adapter; no shadow derivative switch.<span data-glossary-navigation="true"> ([Elementwise glossary](../../../../glossary.md#term-elementwise); [Normalization glossary](../../../../glossary.md#term-normalization); [Convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Pooling glossary](../../../../glossary.md#term-pooling); [Loss glossary](../../../../glossary.md#loss); [Layout glossary](../../../../glossary.md#layout); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Real construction uses V=Tensor directly, not a wrapper per Tensor. The adapter delegates to the
  existing ordinary Tensor methods. Preserve operation order, exact coefficients, canonical saved
  slots, selected input positions, cast-back, special-value/tie behavior and constant binding order.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [canonical glossary](../../../../glossary.md#term-canonical-form); [constant glossary](../../../../glossary.md#term-constant); [binding glossary](../../../../glossary.md#term-binding))</span>
- Preserve the real result-array allocation where practical through a typed array factory, rather
  than generic Object arrays plus copying/wrappers. No runtime/generated-kernel adapter is added.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [kernel glossary](../../../../glossary.md#kernel))</span>
- Reuse FirstOrderAutograd's existing typed constant owner and coefficient definitions. Metadata
  constants retain exact type/bits and explicit literal roles without creating real constant leaves.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Metadata glossary](../../../../glossary.md#term-metadata))</span>
- Metadata construction executes the SAME formula bodies, including eager temporary constructions,
  using ScanBody.describeOrdinary and independently derived CapturedGraphInference descriptors.
  Expose/factor the existing inference dispatch; do not copy it or construct fake nodes/IDs.
  Preserve deferred CropToShape layout compatibility and DISPROVEN/DEFERRED predicate disposition.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [descriptors glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [dispatch glossary](../../../../glossary.md#term-dispatch); [nodes glossary](../../../../glossary.md#node))</span>
- Keep original source-role metadata distinct from lazy resolved primal values. Descriptor access
  alone must not demand a primal. Bind all actual value operands, saved outputs and discrete
  indices/masks through the supplied resolver, retaining original identity contribution keys.<span data-glossary-navigation="true"> ([lazy glossary](../../../../glossary.md#term-lazy-evaluation); [primal glossary](../../../../glossary.md#term-primal); [masks glossary](../../../../glossary.md#term-boolean-mask); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- A synchronous per-created-operation observation seam, for metadata and optional real Tensor
  materialization, may carry future compile-local coordinate/activity facts. It is not another
  semantic graph, registry, persisted certificate, executor, numerical language or Tensor state.
  Cover primal-only generated operations too; ordinary unobserved construction needs no observer allocation.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [materialization glossary](../../../../glossary.md#materialization); [carry glossary](../../../../glossary.md#term-carry); [compile glossary](../../../../glossary.md#compile); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [graph glossary](../../../../glossary.md#graph); [registry glossary](../../../../glossary.md#term-registry); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [observer glossary](../../../../glossary.md#term-observer))</span>
- Reuse existing argument normalization/guards. If the canonical metadata facade lacks a necessary
  shared raw-argument guard, report that exact prerequisite before copying normalization logic.
  No prototype Tensor, reflection into private constructors, sampled B or compatibility shim.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [facade glossary](../../../../glossary.md#term-facade); [guard glossary](../../../../glossary.md#term-guard); [prototype glossary](../../../../glossary.md#term-prototype))</span>
- Intrinsic trailing axes may be named positively in the SAME formula: attention's three SUM
  sites use rank-minus-1/2/3; pooling's two ARG_MAX sites use the candidate's last axis (rank four).
  Preserve identical canonical operations, without copying general negative-axis normalization.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank))</span>
- Stochastic and joint Scan owners remain on their existing paths. Remove obsolete ordinary-only
  overloads/callers after cutover; do not keep a parallel old formula implementation.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

## Files, integration and [validation](../../../../glossary.md#term-validation)

- Existing production owners under modules/compiler/src/main/java/io/github/pho001/synaptik/compiler:
  FirstOrderAutograd, the eleven ordinary *GradientRules owners, CapturedGraphInference and their
  actual shared geometry helpers. One focused package-private adapter owner is permitted if it
  makes responsibility clearer; no Model public API change without a concrete prerequisite review.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [compiler glossary](../../../../glossary.md#term-compiler); [owner glossary](../../../../glossary.md#term-owner); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Depends on: Complete Compiler0017A and Model0046/0047, integrated CPU0078 lock, all verified.
  Exact base: HEAD07a281edd3086efde2853211b71b48288db027e1 plus accepted dirty manifest
  ordinary-gradient-distribution-base.json; MAIN recorded Ready/source ownership before launch.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [gradient glossary](../../../../glossary.md#term-gradient); [distribution glossary](../../../../glossary.md#term-distribution); [Ready glossary](../../../../glossary.md#term-ready); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Conflicts: those Compiler files and their direct focused tests. MAIN owns shared docs/integration.
  Parallel group RELEASE-ORDINARY-GRADIENT-DISTRIBUTION; build/distribution files are disjoint.<span data-glossary-navigation="true"> ([RELEASE glossary](../../../../glossary.md#term-release))</span>
- Order: contract review/Ready -> clean frozen context -> complete factoring -> independent source/
  Javadoc review -> MAIN affected gate and unchanged actual ordinary/lexical VJP consumers -> final
  evidence/docs review and exact integration. Workers run no commands/builds/tests/lint/formatters.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [VJP glossary](../../../../glossary.md#term-vector-jacobian-product-vjp); [run glossary](../../../../glossary.md#term-run))</span>
- Preserve existing behavioral tests; delete obsolete reflection/member/count/source-wording pins,
  never repin them. New permanent tests only for plausible consumer-visible boundaries, not forwarding.
- MAIN proves all existing ordinary scalar-oracle/lexical VJP fixtures unchanged and directly runs
  the checked metadata construction, including typed canonical siblings, invalid construction,
  exact constant roles, zero consumed Tensor IDs and lazy primal observations. Retain first failures.
  Old qualified CPU fingerprints/keys/class bytes must remain unchanged for unchanged recipes.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [metadata glossary](../../../../glossary.md#term-metadata); [canonical glossary](../../../../glossary.md#term-canonical-form); [constant glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [primal glossary](../../../../glossary.md#term-primal); [CPU glossary](../../../../glossary.md#term-cpu))</span>

No new [derivative](../../../../glossary.md#term-derivative) mathematics, new [admission](../../../../glossary.md#term-admission), low numerical unlock, cold/wire expansion, native
work, [runtime](../../../../glossary.md#term-runtime) helper [dispatch](../../../../glossary.md#term-dispatch) or all-[PURE](../../../../glossary.md#term-pure) completion claim. Report exact changed-file preimage/final
SHAs and interface contract; MAIN owns runtime verification, docs, final review and acceptance.

## Result

Each of the eleven existing ordinary formula [owners](../../../../glossary.md#term-owner) now has one generic mathematical body.
The real route uses [Tensor](../../../../glossary.md#tensor) references directly; one package-private OrdinaryConstruction adapts
ordinary operations and exact existing [constants](../../../../glossary.md#term-constant) without wrapping each Tensor. The [metadata](../../../../glossary.md#term-metadata)
route checks the same construction through [Model](../../../../glossary.md#term-model)'s shared guards and the existing independent
[Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference) dispatcher, including [canonical](../../../../glossary.md#term-canonical-form) siblings and deferred CropToShape rules.
Typed result arrays, nullable source positions, [lazy](../../../../glossary.md#term-lazy-evaluation) [primal](../../../../glossary.md#term-primal) [bindings](../../../../glossary.md#term-binding), [eager](../../../../glossary.md#term-eager-evaluation) unused attention V
casting, saved BN/attention/TOP_K slots and constant ingress order remain unchanged. Optional
[synchronous](../../../../glossary.md#term-synchronous) observation adds no event/inference allocation to the unobserved real branch.

MAIN's first build found seven unmigrated Tensor member chains on generic V; the next found the
missing direct NOT_EQUAL adapter mapping used by [pooling](../../../../glossary.md#term-pooling). Both failures are retained. The repairs
delegate to the same ordinary operations, not logical-complement or [derivative](../../../../glossary.md#term-derivative) rewrites. Three
independent source/[Javadoc](../../../../glossary.md#term-javadoc) slices approved the corrected source with no P1/P2 findings, including
the final documented constructor/source-role delta.

The full affected gate passed: Compiler 423, [CPU](../../../../glossary.md#term-cpu) 1,474 and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 566 tests, with 60 existing CPU
opt-in skips. Final Compiler Javadocs/[JAR](../../../../glossary.md#term-jar) passed without the initial three documentation warnings.
The unchanged [scalar](../../../../glossary.md#term-scalar)-[oracle](../../../../glossary.md#term-test-oracle) public forward/actual-backward corpus passed all 84 cases; the
thirty-case indexing subset is a repeated subset, not additional coverage.

All original lexical numerical consumers passed in strict after mode: 18 targeted, 84 ordinary
and 12 [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/pooling cases. Earlier cleanup had removed their throwaway source/classes;
initial missing-class launches are retained. MAIN recovered the exact original source writes
and the original resolved-[layout](../../../../glossary.md#layout)/positive-control patch from this session, reproducing the
historical final [fixture](../../../../glossary.md#term-fixture) [snapshot](../../../../glossary.md#term-snapshot) tags. No oracle/assertion was relaxed. Initial recovery and
incomplete recompile-[classpath](../../../../glossary.md#term-classpath) failures remain separate from the successful final replay.

A direct checked-construction [CLI](../../../../glossary.md#term-cli) exercised all eleven ordinary formula families over 100 real
source fixtures: 112 metadata [VJP](../../../../glossary.md#term-vector-jacobian-product-vjp) calls produced 184 normalized positional contributions, consuming
zero Tensor [identities](../../../../glossary.md#term-identity-meanings). Focused permanent behavior tests additionally cover exact literal bits
and binding [identity](../../../../glossary.md#term-identity-meanings), complete canonical saved-sibling rejection, independent deferred predicates,
Crop layout compatibility and real/metadata eager observation parity.
The unchanged fourteen-family/twenty-eight-[kernel](../../../../glossary.md#kernel) [capture](../../../../glossary.md#term-capture-meanings) retained all 85 identity/compatibility/
class/index files byte-for-byte.

Exact source and evidence: `compiler0017b-main-integration.json`,
`compiler0017b-runtime-artifact-evidence.json`, `compiler0017b-metadata-construction-evidence.json`
and retained failed/successful gate [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) in the production-[run](../../../../glossary.md#term-run) directory. This is ordinary
construction parity, not a [finite](../../../../glossary.md#term-finite) selected-demand proof, structural [cotangent](../../../../glossary.md#cotangent) activity plan,
typed [Scan](../../../../glossary.md#term-cumulative-scan) [AD](../../../../glossary.md#autograd), physical composition, cold/wire [widening](../../../../glossary.md#term-widening), low-compute unlock or complete [release](../../../../glossary.md#term-release).
