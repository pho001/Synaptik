# Task 0048: Shared per-output ordinary Scan value semantics

## Status

Complete — MAIN integrated fourteen exact source/test paths after independent mapping, source-
boundary and consumer reviews, affected gates and actual public metadata/cold consumers.
Final independent source/evidence/documentation APPROVE0; no wider typed AD, backend or wire grant.

## Goal and owner

Class C. Replace the duplicated producer-wide length-Scan role algebra with one descriptor-only
per-output semantic seam on existing model.tensor.Scan. Ordinary Model guards and independent
Compiler inference remain distinct. This is the source/coordinate prerequisite of the mandatory
all-PURE composition target, not a narrower final mathematical inventory.

Read the root scope index, the Scan owner's Header/bindings/source admission, Bounded row-role
analysis and Activity/raw identity/retained histories, the foundational Model owner, and Compiler
AD/cold boundaries. The coordinated source-coordinate authority is accepted; ADR0061's wider
typed-demand/activity/physical continuation remains proposed, not a blanket implementation grant.

## Shared interface

In existing Scan:

```java
public record ValueSemantics(OptionalInt rowAxis, boolean terminal, List<Integer> directInputs) {}
public static ValueSemantics declaredSemantics(
        TensorDescriptor descriptor, OptionalInt rowAxis, Dimension batch);
public static List<ValueSemantics> ordinarySemantics(Operation operation,
        List<TensorDescriptor> inputs, List<ValueSemantics> inputSemantics,
        List<TensorDescriptor> outputs, Dimension batch);
```

- SHARED has no row axis. ROW(axis) identifies exactly one original B coordinate at any valid
  rank/axis. Terminal implies SHARED. Direct input positions are immutable, increasing and unique;
  tied operands preserve their separate positions. They are value/control routes, not cotangent
  eligibility or a union of descriptors consulted for shape/promotion.
- Declared facts validate the exact static or named B at the explicit axis, static other axes and
  checked fixed-feature products, including unused declarations, canonical siblings and T/B=0/1.
  Equal extents never infer a role. No sampled B, prototype Tensor, ID or CPU geometry.
- Ordinary semantics consumes independently checked canonical descriptors, not a persisted/source
  certificate. It derives no ordinary output shape/promotion and allocates no Tensor/producer/ID.
  Unknown/custom/effectful/RNG/dropout/fixed/nested operations remain excluded even when all SHARED.
- Known all-SHARED PURE families have no row-coordinate kind blacklist. For ROW operations, prove
  that each contributor reads the same identified row; feature traversal is ordinary. Preserve
  reshape boundaries using existing exact coordinate checks, not total-size/zero/singleton guesses.
  Rank edits/permutation map explicit axes; feature slicing/order/cumulative/normalization/windows/
  lookup preserve ROW only through their actual ordinary contributor mapping.
- Produce all canonical siblings separately. BatchNorm training routes are slot0:[0,1,2],
  slot1:[0,3], slot2:[0,4], slots3/4:[0]. Attention weights depend on Q/K/optional mask, not V;
  output depends on Q/K/V/optional mask. Batch domain can arise from shape broadcasting separately
  from value dependence. Preserve exact typed/discrete canonical outputs and ordinary predicates.
- ROW next-carry/emission boundaries stay ROW(0), nonterminal. Intermediates may have any proved
  row axis/rank, including grouped Conv2d VJP [B,G,O,P] -> [G,B,O,P] -> batched MATMUL -> SUM(axis1).
- Only established active terminal meanings eliminate ROW: represented SUM/sumToShape and matched
  MATMUL contraction, plus functional shared-target ADD-scatter with same-row indices/updates,
  ordinary update order, base exactly once and complete active index prevalidation. No inferred
  active MEAN/variance/softmax/order/loss divisor or replacement/MUL terminal meaning.
- Terminal taint follows per-output value routes and never feeds ROW. A terminal shared ADD-scatter
  base is permitted only at shared targets, not as terminal-derived row updates/indices. Every
  syntactic shared-carry use still obeys exact own left old.add(contribution), matching shape/type,
  no contribution dependence on any shared carry and a distinct represented per-time ADD.

## Cutover and independent boundaries

Model Scan.validateRows, raw Compiler ScanBodyInference.validateRows and derivative role discovery
in ScanGradientRules use the same seam after their own descriptor validation. Remove the old
rowResult implementations and migrate every caller/test oracle; no alias or shadow role algebra.
Check shared-carry syntactic isolation separately from per-output numeric dependencies.

This source/semantic cutover does not silently qualify new AD, Prepare/CPU or cold/wire recipes.
Raw inference retains nonempty/positive-ROW topology, homogeneous F32/F64 and declared-header
restrictions; its independently checked forward coordinate admission intentionally broadens under
the shared semantic criterion. Existing AD, Prepare/CPU and cold/wire operation/rank/attribute
domains and original v1-v5 selection remain separate owning predicates.
Where old cold scope relied implicitly on the old role helper, make its existing admission predicate
explicit using the new facts plus old representability limits, not a copied old transfer algorithm.
Reuse the existing cold preflight owner across consumers where applicable; no fake source Tensors.
Sequence-header axis widening for newly generated histories requires its own coordinated typed
consumer/codec guards; arbitrary intermediate axes alone do not waive current header admission.

## Integration and acceptance

- Depends on Complete Model0046/0047, Compiler0017A and CPU0078. Wider typed VJP proof/presence and
  ordinary CPU composition remain separate coordinated prerequisites, not implicit support.
- Source owners: Model Scan.java/ScanBody.java; Compiler ScanBodyInference.java/ScanGradientRules.java/
  SourceGraphReconstruction.java; existing codec cold hook only if required to preserve its effective
  old intersection. Migrate the direct ScanGradientRulesTest role oracle and affected genuine
  Model/Compiler/Checkpoint/Prepare/CPU behavior fixtures; no source-text/count/wiring tests.
- Existing TensorWindowExpressions.windowOutputDimension/windowOutputSize may become package-private
  with meaningful shared-owner Javadoc so exact flattened-window coordinate proofs reuse ordinary
  checked geometry. Keep both algorithms and their existing callers unchanged; no duplicate formula.
- Conflicts: those source owners, disjoint from Compiler0017B ordinary owners/central inference
  dispatcher and CPU0080 generator/loss emitter. MAIN owns shared contracts/docs and integration.
- Parallel group RELEASE-SCAN-COORDINATE-METHODS. MAIN records the exact accepted dirty-source
  manifest and two clean contexts before source launch; no build/test work is delegated.
- One exact dirty-source clean context, independent contract/source/Javadoc review, MAIN affected
  gates and actual public metadata/source calls, then final evidence/docs acceptance. Workers run
  no commands/builds/tests/lint/formatters/native loads.
- Prove each supported family's genuine coordinate cases and mixed-role rejection, all-SHARED
  families, output-specific BatchNorm/attention dependencies, nonleading grouped-convolution paths,
  terminal embedding scatter, B=0/1/named, overflow and pre-ID error ordering. Preserve raw old
  consumer rejects and actual retained CPU artifacts/wire bytes; retain first failures truthfully.

No new semantic graph, registry, derivative mathematics, executor, numerical mode, low compute
unlock or universal backend support. Final all-PURE acceptance still requires actual generated
first-order backward and existing-owner physical execution across the complete declared domain.

## Result

Existing Scan owns the three public descriptor-only APIs above. Model construction, independent
raw inference, derivative bookkeeping and the genuine expression oracle share one per-canonical-
output semantic transfer; obsolete producer-wide rowResult implementations are removed. Known
entirely SHARED PURE families no longer need a coordinate-kind whitelist. Arbitrary intermediate
ROW axes/ranks use complete ordinary contributor mappings, including grouped rank-four MATMUL,
terminal SUM and functional shared-target ADD-scatter. BatchNorm's five slots and attention's
output/weights retain separate direct value/control routes and shape-provided batch domains.
Syntactic shared-carry isolation, exact accumulator type/shape and ROW(0) roots remain independent.

The original strict-uniqueness proposal would regress `[B,F] -> [B,1,F] -> SUM(axis1)` at B=1.
MAIN obtained independent approval for the coordinated complete-witness amendment: each candidate
must prove canonical B and both checked feature boundaries; any fully proved witness suffices,
with lowest-axis selection and exact-shape identity. There is no singleton/zero/extent-only bypass.
FOLD uses the same existential contributor criterion and existing ordinary window-count owner.
Only package-private visibility/Javadoc changed in that ordinary geometry helper; no duplicate math.

The first affected gate failed on record-class Javadoc tags, a bad singleton-FOLD negative
fixture and a changed cold overflow exception boundary. Review additionally found omitted untouched
scatter update-axis mappings and a late AD history/header rejection. All failures/findings are
retained. Corrections keep the valid orthogonal FOLD witness, add a genuine all-candidate-mixing
negative, follow the complete ordinary scatter coordinate mapping, and preserve contextual cold
IllegalArgumentException with its arithmetic cause. The metadata API's checked overflow remains.

Newly source-valid rank-three ROW(1) TANH history was actually accepted by preflight before the
repair in both F32/F64; afterward it rejects with zero derivative Tensor identities. The existing
semantic collector is reused and its map shared with expansion, rather than recomputed. This
retains the old independent AD/header representation before derivative allocation; it is not a
new AD admission. Complete lexical value typing and selected-demand/activity proof remain later work.

The recovered full affected gate passed Model 1,182, Compiler 423, Checkpoint 81, Prepare 100 and
CPU 1,475 tests: 3,261 passes with 60 existing CPU opt-in skips. Model/Compiler Javadocs and
consumer library builds passed. A public CLI checked 96 ordinary/window all-SHARED source cases,
reshape B=0/1/3/named, grouped ROW(1) rank-four contraction, canonical BN/attention routes,
untouched-axis scatter and empty-batch feature overflow without metadata Tensor IDs.

The identical fourteen-family/twenty-eight-kernel capture retains all 85 old generated identity/
compatibility/class/index files byte-for-byte. An unchanged public GraphCheckpoint save/load/rebind
program passed 48 F32/F64 all-valid/length cases, B=0/1/3, T=0/2 and both directions. Canonical
reconstructed graphs and every v4/v5 archive byte match the installed pre-change publications.
This is exact proof for those cases, not an exhaustive wire-inventory assertion; existing v1-v5
behavior fixtures also passed. Initial missing-JAR/import smoke failures are retained separately.

Cold reconstruction explicitly retains its former operation/rank/role/attribute intersection;
SHARED RELU and batched MATMUL forward metadata are not thereby cold/AD/CPU support. Original
sequence headers stay axes 0/1 and rank-two ROW(1); direct raw/Prepare/CPU negatives still pass.
No codec production change, new format/opcode, numerical policy, low unlock or physical PURE
composition was introduced.

Exact source and evidence: `model0048-main-integration.json`, `model0048-consumer-evidence.json`
and `model0048-wire-retention-evidence.json` in the production-run directory. Public API usage is
documented in [Tensor metadata](../../../../api/tensor-api.md#per-output-ordinary-coordinate-facts).
Full all-PURE generated backward/physical execution and final release qualification remain open.
