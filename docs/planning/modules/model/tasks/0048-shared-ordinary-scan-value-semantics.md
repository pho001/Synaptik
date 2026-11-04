# Task 0048: Shared per-output ordinary [Scan](../../../../glossary.md#term-cumulative-scan) value semantics

## Status

Complete — MAIN integrated fourteen exact source/test paths after independent mapping, source-
boundary and consumer reviews, affected gates and actual public [metadata](../../../../glossary.md#term-metadata)/cold consumers.
Final independent source/evidence/documentation APPROVE0; no wider typed [AD](../../../../glossary.md#autograd), [backend](../../../../glossary.md#backend) or wire grant.

## Goal and [owner](../../../../glossary.md#term-owner)

Class C. Replace the duplicated [producer](../../../../glossary.md#term-producer)-wide length-[Scan](../../../../glossary.md#term-cumulative-scan) role algebra with one [descriptor](../../../../glossary.md#term-descriptor)-only
per-output semantic seam on existing [model](../../../../glossary.md#term-model).[tensor](../../../../glossary.md#tensor).Scan. Ordinary Model guards and independent
[Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference) remain distinct. This is the source/[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) prerequisite of the mandatory
all-[PURE](../../../../glossary.md#term-pure) composition [target](../../../../glossary.md#term-target), not a narrower final mathematical inventory.

Read the root [scope](../../../../glossary.md#term-scope) index, the Scan owner's Header/[bindings](../../../../glossary.md#term-binding)/source [admission](../../../../glossary.md#term-admission), Bounded row-role
analysis and Activity/raw [identity](../../../../glossary.md#term-identity-meanings)/retained histories, the foundational Model owner, and Compiler
[AD](../../../../glossary.md#autograd)/cold boundaries. The coordinated source-coordinate authority is accepted; ADR0061's wider
typed-demand/activity/physical continuation remains proposed, not a blanket implementation grant.

## Shared interface

In existing [Scan](../../../../glossary.md#term-cumulative-scan):

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
  eligibility or a union of descriptors consulted for shape/promotion.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank glossary](../../../../glossary.md#term-rank); [immutable glossary](../../../../glossary.md#term-immutable); [cotangent glossary](../../../../glossary.md#cotangent); [descriptors glossary](../../../../glossary.md#term-descriptor); [shape glossary](../../../../glossary.md#shape))</span>
- Declared facts validate the exact static or named B at the explicit axis, static other axes and
  checked fixed-feature products, including unused declarations, canonical siblings and T/B=0/1.
  Equal extents never infer a role. No sampled B, prototype Tensor, ID or CPU geometry.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [canonical glossary](../../../../glossary.md#term-canonical-form); [prototype glossary](../../../../glossary.md#term-prototype); [Tensor glossary](../../../../glossary.md#tensor); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Ordinary semantics consumes independently checked canonical descriptors, not a persisted/source
  certificate. It derives no ordinary output shape/promotion and allocates no Tensor/producer/ID.
  Unknown/custom/effectful/RNG/dropout/fixed/nested operations remain excluded even when all SHARED.<span data-glossary-navigation="true"> ([certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [producer glossary](../../../../glossary.md#term-producer); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Known all-SHARED PURE families have no row-coordinate kind blacklist. For ROW operations, prove
  that each contributor reads the same identified row; feature traversal is ordinary. Preserve
  reshape boundaries using existing exact coordinate checks, not total-size/zero/singleton guesses.
  Rank edits/permutation map explicit axes; feature slicing/order/cumulative/normalization/windows/
  lookup preserve ROW only through their actual ordinary contributor mapping.<span data-glossary-navigation="true"> ([PURE glossary](../../../../glossary.md#term-pure); [singleton glossary](../../../../glossary.md#term-singleton); [permutation glossary](../../../../glossary.md#permutation); [normalization glossary](../../../../glossary.md#term-normalization); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Produce all canonical siblings separately. BatchNorm training routes are slot0:[0,1,2],
  slot1:[0,3], slot2:[0,4], slots3/4:[0]. Attention weights depend on Q/K/optional mask, not V;
  output depends on Q/K/V/optional mask. Batch domain can arise from shape broadcasting separately
  from value dependence. Preserve exact typed/discrete canonical outputs and ordinary predicates.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [mask glossary](../../../../glossary.md#term-boolean-mask); [Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- ROW next-carry/emission boundaries stay ROW(0), nonterminal. Intermediates may have any proved
  row axis/rank, including grouped Conv2d VJP [B,G,O,P] -> [G,B,O,P] -> batched MATMUL -> SUM(axis1).<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [VJP glossary](../../../../glossary.md#term-vector-jacobian-product-vjp); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Only established active terminal meanings eliminate ROW: represented SUM/sumToShape and matched
  MATMUL contraction, plus functional shared-target ADD-scatter with same-row indices/updates,
  ordinary update order, base exactly once and complete active index prevalidation. No inferred
  active MEAN/variance/softmax/order/loss divisor or replacement/MUL terminal meaning.<span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [target glossary](../../../../glossary.md#term-target); [MEAN glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [loss glossary](../../../../glossary.md#loss))</span>
- Terminal taint follows per-output value routes and never feeds ROW. A terminal shared ADD-scatter
  base is permitted only at shared targets, not as terminal-derived row updates/indices. Every
  syntactic shared-carry use still obeys exact own left old.add(contribution), matching shape/type,
  no contribution dependence on any shared carry and a distinct represented per-time ADD.

## Cutover and independent boundaries

[Model](../../../../glossary.md#term-model) [Scan](../../../../glossary.md#term-cumulative-scan).validateRows, raw [Compiler](../../../../glossary.md#term-compiler) ScanBodyInference.validateRows and [derivative](../../../../glossary.md#term-derivative) role discovery
in ScanGradientRules use the same seam after their own [descriptor](../../../../glossary.md#term-descriptor) [validation](../../../../glossary.md#term-validation). Remove the old
rowResult implementations and migrate every caller/[test oracle](../../../../glossary.md#term-test-oracle); no [alias](../../../../glossary.md#aliasing-and-arena) or shadow role algebra.
Check shared-[carry](../../../../glossary.md#term-carry) syntactic isolation separately from per-output numeric dependencies.

This source/semantic cutover does not silently qualify new [AD](../../../../glossary.md#autograd), [Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu) or cold/wire recipes.
Raw [inference](../../../../glossary.md#term-inference) retains nonempty/positive-ROW [topology](../../../../glossary.md#term-topology), [homogeneous](../../../../glossary.md#term-homogeneous) [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) and declared-header
restrictions; its independently checked forward [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [admission](../../../../glossary.md#term-admission) intentionally broadens under
the shared semantic criterion. Existing AD, Prepare/CPU and cold/wire [operation](../../../../glossary.md#operation)/[rank](../../../../glossary.md#term-rank)/attribute
domains and original v1-v5 selection remain separate owning predicates.
Where old cold [scope](../../../../glossary.md#term-scope) relied implicitly on the old role helper, make its existing admission predicate
explicit using the new facts plus old representability limits, not a copied old transfer algorithm.
Reuse the existing cold [preflight](../../../../glossary.md#term-preflight) [owner](../../../../glossary.md#term-owner) across consumers where applicable; no fake source [Tensors](../../../../glossary.md#tensor).
Sequence-header [axis](../../../../glossary.md#term-axis) [widening](../../../../glossary.md#term-widening) for newly generated histories requires its own coordinated typed
consumer/[codec](../../../../glossary.md#term-codec) guards; arbitrary intermediate axes alone do not waive current header admission.

## Integration and acceptance

- Depends on Complete Model0046/0047, Compiler0017A and CPU0078. Wider typed VJP proof/presence and
  ordinary CPU composition remain separate coordinated prerequisites, not implicit support.<span data-glossary-navigation="true"> ([VJP glossary](../../../../glossary.md#term-vector-jacobian-product-vjp); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Source owners: Model Scan.java/ScanBody.java; Compiler ScanBodyInference.java/ScanGradientRules.java/
  SourceGraphReconstruction.java; existing codec cold hook only if required to preserve its effective
  old intersection. Migrate the direct ScanGradientRulesTest role oracle and affected genuine
  Model/Compiler/Checkpoint/Prepare/CPU behavior fixtures; no source-text/count/wiring tests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [Compiler glossary](../../../../glossary.md#term-compiler); [codec glossary](../../../../glossary.md#term-codec); [oracle glossary](../../../../glossary.md#term-test-oracle); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Existing TensorWindowExpressions.windowOutputDimension/windowOutputSize may become package-private
  with meaningful shared-owner Javadoc so exact flattened-window coordinate proofs reuse ordinary
  checked geometry. Keep both algorithms and their existing callers unchanged; no duplicate formula.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Conflicts: those source owners, disjoint from Compiler0017B ordinary owners/central inference
  dispatcher and CPU0080 generator/loss emitter. MAIN owns shared contracts/docs and integration.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [loss glossary](../../../../glossary.md#loss))</span>
- Parallel group RELEASE-SCAN-COORDINATE-METHODS. MAIN records the exact accepted dirty-source
  manifest and two clean contexts before source launch; no build/test work is delegated.<span data-glossary-navigation="true"> ([RELEASE glossary](../../../../glossary.md#term-release); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- One exact dirty-source clean context, independent contract/source/Javadoc review, MAIN affected
  gates and actual public metadata/source calls, then final evidence/docs acceptance. Workers run
  no commands/builds/tests/lint/formatters/native loads.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [run glossary](../../../../glossary.md#term-run))</span>
- Prove each supported family's genuine coordinate cases and mixed-role rejection, all-SHARED
  families, output-specific BatchNorm/attention dependencies, nonleading grouped-convolution paths,
  terminal embedding scatter, B=0/1/named, overflow and pre-ID error ordering. Preserve raw old
  consumer rejects and actual retained CPU artifacts/wire bytes; retain first failures truthfully.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [embedding glossary](../../../../glossary.md#embedding); [overflow glossary](../../../../glossary.md#term-overflow); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

No new semantic [graph](../../../../glossary.md#graph), [registry](../../../../glossary.md#term-registry), [derivative](../../../../glossary.md#term-derivative) mathematics, executor, numerical mode, low compute
unlock or universal [backend](../../../../glossary.md#backend) support. Final all-[PURE](../../../../glossary.md#term-pure) acceptance still requires actual generated
first-order backward and existing-[owner](../../../../glossary.md#term-owner) physical execution across the complete declared domain.

## Result

Existing [Scan](../../../../glossary.md#term-cumulative-scan) owns the three public [descriptor](../../../../glossary.md#term-descriptor)-only APIs above. [Model](../../../../glossary.md#term-model) construction, independent
raw [inference](../../../../glossary.md#term-inference), [derivative](../../../../glossary.md#term-derivative) bookkeeping and the genuine expression [oracle](../../../../glossary.md#term-test-oracle) share one per-[canonical](../../../../glossary.md#term-canonical-form)-
output semantic transfer; obsolete [producer](../../../../glossary.md#term-producer)-wide rowResult implementations are removed. Known
entirely SHARED [PURE](../../../../glossary.md#term-pure) families no longer need a [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-kind whitelist. Arbitrary intermediate
ROW [axes](../../../../glossary.md#term-axis)/ranks use complete ordinary contributor mappings, including grouped [rank](../../../../glossary.md#term-rank)-four [MATMUL](../../../../glossary.md#matrix-multiplication--matmul),
terminal SUM and functional shared-[target](../../../../glossary.md#term-target) ADD-scatter. BatchNorm's five slots and attention's
output/[weights](../../../../glossary.md#term-weight) retain separate direct value/control routes and [shape](../../../../glossary.md#shape)-provided [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) domains.
Syntactic shared-[carry](../../../../glossary.md#term-carry) isolation, exact [accumulator type](../../../../glossary.md#accumulator-type)/shape and ROW(0) roots remain independent.

The original strict-uniqueness proposal would regress `[B,F] -> [B,1,F] -> SUM(axis1)` at B=1.
MAIN obtained independent approval for the coordinated complete-witness amendment: each candidate
must prove canonical B and both checked [feature](../../../../glossary.md#term-feature) boundaries; any fully proved witness suffices,
with lowest-axis selection and exact-shape [identity](../../../../glossary.md#term-identity-meanings). There is no [singleton](../../../../glossary.md#term-singleton)/zero/[extent](../../../../glossary.md#term-extent)-only bypass.
FOLD uses the same existential contributor criterion and existing ordinary window-count [owner](../../../../glossary.md#term-owner).
Only package-private visibility/[Javadoc](../../../../glossary.md#term-javadoc) changed in that ordinary geometry helper; no duplicate math.

The first affected gate failed on record-class Javadoc tags, a bad singleton-FOLD negative
[fixture](../../../../glossary.md#term-fixture) and a changed cold [overflow](../../../../glossary.md#term-overflow) exception boundary. Review additionally found omitted untouched
scatter update-axis mappings and a late [AD](../../../../glossary.md#autograd) history/header rejection. All failures/findings are
retained. Corrections keep the valid orthogonal FOLD witness, add a genuine all-candidate-mixing
negative, follow the complete ordinary scatter coordinate mapping, and preserve contextual cold
IllegalArgumentException with its arithmetic cause. The [metadata](../../../../glossary.md#term-metadata) [API](../../../../glossary.md#term-application-programming-interface-api)'s checked overflow remains.

Newly source-valid rank-three ROW(1) [TANH](../../../../glossary.md#term-tanh) history was actually accepted by [preflight](../../../../glossary.md#term-preflight) before the
repair in both [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding); afterward it rejects with zero derivative [Tensor](../../../../glossary.md#tensor) identities. The existing
semantic collector is reused and its map shared with expansion, rather than recomputed. This
retains the old independent AD/header representation before derivative allocation; it is not a
new AD [admission](../../../../glossary.md#term-admission). Complete lexical value typing and selected-demand/activity proof remain later work.

The recovered full affected gate passed Model 1,182, [Compiler](../../../../glossary.md#term-compiler) 423, [Checkpoint](../../../../glossary.md#term-checkpoint) 81, [Prepare](../../../../glossary.md#prepare) 100 and
[CPU](../../../../glossary.md#term-cpu) 1,475 tests: 3,261 passes with 60 existing CPU opt-in skips. Model/Compiler Javadocs and
consumer library builds passed. A public [CLI](../../../../glossary.md#term-cli) checked 96 ordinary/window all-SHARED source cases,
reshape B=0/1/3/named, grouped ROW(1) rank-four [contraction](../../../../glossary.md#contraction-dimension), canonical BN/attention routes,
untouched-axis scatter and empty-batch feature overflow without metadata Tensor IDs.

The identical fourteen-family/twenty-eight-[kernel](../../../../glossary.md#kernel) [capture](../../../../glossary.md#term-capture-meanings) retains all 85 old generated identity/
compatibility/class/index files byte-for-byte. An unchanged public GraphCheckpoint save/load/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction)
program passed 48 F32/F64 all-valid/length cases, B=0/1/3, T=0/2 and both directions. Canonical
reconstructed [graphs](../../../../glossary.md#graph) and every v4/v5 archive byte match the installed pre-change publications.
This is exact proof for those cases, not an exhaustive wire-inventory assertion; existing v1-v5
behavior fixtures also passed. Initial missing-[JAR](../../../../glossary.md#term-jar)/import [smoke](../../../../glossary.md#term-smoke-test) failures are retained separately.

[Cold reconstruction](../../../../glossary.md#term-cold-reconstruction) explicitly retains its former [operation](../../../../glossary.md#operation)/rank/role/attribute intersection;
SHARED RELU and batched MATMUL forward metadata are not thereby cold/AD/CPU support. Original
sequence headers stay axes 0/1 and [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1); direct raw/Prepare/CPU negatives still pass.
No [codec](../../../../glossary.md#term-codec) production change, new format/[opcode](../../../../glossary.md#term-opcode), numerical policy, low unlock or physical PURE
composition was introduced.

Exact source and evidence: `model0048-main-integration.json`, `model0048-consumer-evidence.json`
and `model0048-wire-retention-evidence.json` in the production-[run](../../../../glossary.md#term-run) directory. Public API usage is
documented in [Tensor metadata](../../../../api/tensor-api.md#per-output-ordinary-coordinate-facts).
Full all-PURE generated backward/physical execution and final [release](../../../../glossary.md#term-release) [qualification](../../../../glossary.md#term-qualification) remain open.
