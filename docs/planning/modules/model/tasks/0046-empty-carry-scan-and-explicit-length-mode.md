# Task 0046: Empty-[carry](../../../../glossary.md#term-carry) Scan and explicit length-mode source

**Editorial reading key (current context, not a status update).**

This record documents a source-representation cutover, not execution enablement. Callers declare
step placeholders (formals) in `ScanBody`; `Scan.apply` binds actual Tensor references by position
and returns canonical output descriptions, without reading values or running the body. Carry passes
state between steps; emissions collect step outputs. Empty carry therefore permits an emission-only
scan, not an empty body. The [source-binding
contract](../../../../reference/tensor/general-scan.md#formals-actual-positions-and-source-closure)
explains these positions and required roots. Model defines meaning; providers realize independently
supported routes, not redefine it. Semantic responsibility does not imply buffer ownership:
storage-free expressions own no result buffers.

Explicit `lengthAware` distinguishes length mode from all-valid mode even with zero ROW carries. ROW
identifies a proved batch coordinate; SHARED has none. For example, an emission-only length scan
still receives lengths last, with no lengths formal; its batch comes from the lengths descriptor,
not a first carry. The [length-mode
contract](../../../../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution)
explains these roles and separate consumer admission. Its current descriptions do not enlarge this
task’s historical compiler, execution or frozen-wire scope.

**Emission-only reading illustration.** Over finite real scalars, take step inputs $x_0=2$, $x_1=3$
and an emission root $e_t=2x_t$ for $t\in\{0,1\}$. With no carry slots, stacked emissions are
$(4,6)$ and the final-carry list is empty. There is still a body and time traversal; empty carry
does not mean no outputs or that length mode can be inferred from a carry count. The example is
mathematical, not evidence of execution admission.

## Status

Complete for bounded source only — exact reviewed source/caller/[guard](../../../../glossary.md#term-guard) and sole-formatter cutover
is integrated. Affected3,998passes/60existing [CPU](../../../../glossary.md#term-cpu) skips, actual96source/reject cases and28old CPU
[artifact](../../../../glossary.md#term-artifact) checks passed. Independent final core and wire/evidence/docs reviews APPROVE0.
Typed execution/[AD](../../../../glossary.md#autograd), all-[PURE](../../../../glossary.md#term-pure) composition and [persistence](../../../../glossary.md#term-persistence) are not expanded.

## Change class

Class C — advanced public [scalar](../../../../glossary.md#term-scalar)-header migration and independent cross-[module](../../../../glossary.md#term-module) consumer guards.
Clean implementation, independent source/[Javadoc](../../../../glossary.md#term-javadoc) review and MAIN source/compatibility [smoke](../../../../glossary.md#term-smoke-test)/gates.

## Goal and source evidence

Represent an actual empty [carry](../../../../glossary.md#term-carry), and length-aware reverse scans with zero ROW carries, in the
existing Scan. Before this cutover, ScanBody forbade empty carry and ScanAttrs required carryCount>=1
and used rowCarryCount=0 as all-valid. A legal first-order [derivative](../../../../glossary.md#term-derivative) with only discrete original
carries can need no floating reverse carry, or only SHARED floating accumulators. Do not introduce dummy
floating state, fake discrete [gradients](../../../../glossary.md#term-gradient), replay, [unrolling](../../../../glossary.md#term-unrolling), a map namespace or a second executor.

The [reviewed proposal](../../../../design/decisions/0059-empty-carry-scan-and-explicit-length-mode.md)
uses established fixed-[shape](../../../../glossary.md#shape)/pytree practice, not copied framework code. This task delivers source
representation and truthful independent [guard](../../../../glossary.md#term-guard) preservation, not typed [inference](../../../../glossary.md#term-inference)/[AD](../../../../glossary.md#autograd)/[CPU](../../../../glossary.md#term-cpu) enablement.

## Exact source cutover

- Existing ScanBody permits empty carry and nextCarry together. Step sequences stay nonempty;
  carry+emission roots stay nonempty. Own nonempty carry shape/dtype equality is unchanged.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [shape glossary](../../../../glossary.md#shape); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>
- Existing ScanAttrs [canonical](../../../../glossary.md#term-canonical-form) components become `(timeCount, carryCount, sequenceCount,
  invariantCount, emissionCount, residualCount, reverse, lengthAware, rowCarryCount,
  sequenceRowAxes)`. Preserve the meaningful seven-argument all-valid convenience constructor;
  remove/migrate the old nine-argument sentinel form everywhere, with no shim/alias.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Checked counts: T>=0, c>=0, s>=1, i/e/r>=0 and c+e>=1. All-valid means explicit false, rowCarryCount=0,
  empty axes; length-aware true allows rowCarryCount in [0,c] and one existing optional role per
  sequence. inputCount uses lengthAware, never the row count; lengths stays last and has no formal.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>
- Existing Scan.apply overloads accept the matching empty actual list. Infer static/named B from
  validLengths INT64[B] metadata before checking every explicit ROW declaration. No first-carry
  assumption, length read, extent-based row guess or altered role/algebra rule. Existing positive-
  ROW source meanings, axes, aliases, canonical ports, flags and storage-free validation stay.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype); [metadata glossary](../../../../glossary.md#term-metadata); [extent glossary](../../../../glossary.md#term-extent); [validation glossary](../../../../glossary.md#term-validation))</span>
- ScanKind minimum inputs becomes one for carry-free all-valid with one sequence. CompiledScanBody
  structural admission permits empty nextCarry but requires at least one carry/emission root.
  ScanResult already permits empty groups; preserve it without another result type.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>

Migrate all header-derived sentinel checks and constructors. Do not mechanically rewrite [CPU](../../../../glossary.md#term-cpu) [IR](../../../../glossary.md#term-intermediate-representation-ir)'s
separate qualified internal fields or broaden a downstream gate just because a source flag exists.
No added/moved production type/package/[dependency](../../../../glossary.md#term-dependency)/[callback](../../../../glossary.md#term-callback)/[operation](../../../../glossary.md#operation) kind or [runtime](../../../../glossary.md#term-runtime) protocol.

## Independent consumer and frozen-wire boundaries

[Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference)/[AD](../../../../glossary.md#autograd), direct [Prepare](../../../../glossary.md#prepare) and direct [CPU](../../../../glossary.md#term-cpu) must reject both newly representable forms:
`carryCount==0` OR `lengthAware && rowCarryCount==0`, before first-[carry](../../../../glossary.md#term-carry) assumptions, [derivative](../../../../glossary.md#term-derivative) IDs,
analyzer calls, resources or generated realization. Keep ordinary [homogeneous](../../../../glossary.md#term-homogeneous) nonempty accepted
cases and independently authenticate source/header/body cardinalities. Structural CompiledScanBody
validity is not inference/execution permission. Later consumers explicitly qualify these forms.

Cold SourceGraphReconstruction and raw GraphArtifactCodec retain the old intersection independently.
Opcode23/v4 still requires nonempty carry; opcode33/v5 requires nonempty carry and positive ROW
carry. Writer rejects before sizing/storage, reader before body/role allocations where possible.
Mode is [opcode](../../../../glossary.md#term-opcode)-derived; old seven-field tuples, roles, opcodes, v1-v5 selection and accepted bytes
stay unchanged. No [boolean](../../../../glossary.md#data-type--datatype) wire, v6, invented opcode or silent empty-carry [serialization](../../../../glossary.md#term-serialization). This
includes raw codecs after relaxed ScanKind/ScanAttrs structural checks; early Compiler failure
alone is not [codec](../../../../glossary.md#term-codec) evidence. [Checkpoint](../../../../glossary.md#term-checkpoint) copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) acquisition stay in their accepted bounds.

Compiler0017's obsolete no-floating-carry rejection has been removed from its Draft proposal and
acceptance/dependencies now require this representation. This task does not implement that reverse;
ordinary floating [target](../../../../glossary.md#term-target)/cast/disconnected/unsupported-[operation](../../../../glossary.md#operation) and higher-order gates stay.

## Contracts and authority prerequisites

- [Root](../../../../../ARCHITECTURE.md#core-lifecycle): lifecycle, source-only status and exact
  per-owner consumer gates. Before Ready MAIN coordinates explicit header/source permission.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner); [Ready glossary](../../../../glossary.md#term-ready))</span>
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  source/header/bindings, canonical outputs, traversal/empties, accepted per-value typing and row
  semantics. Source counts/mode and independent downstream gates must change together.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
  and its bounded cold reconstruction; <span data-glossary-navigation="true"> ([cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>[Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  and [CPU](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>: retain negative [admission](../../../../glossary.md#term-admission).
- [Checkpoint](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  retains old v4/v5 domain/bytes. The source flag is not a wire allocation or CPU identity bump.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Files and current [owners](../../../../glossary.md#term-owner)

[Model](../../../../glossary.md#term-model): [operation](../../../../glossary.md#operation).scan.ScanAttrs/ScanKind; [tensor](../../../../glossary.md#tensor).Scan/ScanBody/ScanResult Javadocs; [graph](../../../../glossary.md#graph).CompiledScanBody;
focused constructor/attachment/body/semantics tests. No new production type or helper protocol.

All old [canonical](../../../../glossary.md#term-canonical-form) constructor and header-derived mode callers/tests across [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning),
[Prepare](../../../../glossary.md#prepare), [CPU](../../../../glossary.md#term-cpu), [Checkpoint](../../../../glossary.md#term-checkpoint), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and test/tool fixtures must migrate. Preserve current [ABI](../../../../glossary.md#term-application-binary-interface-abi)/[IR](../../../../glossary.md#term-intermediate-representation-ir) bytes of
qualified CPU paths; only source header access changes. SourceGraphReconstruction, GraphArtifactCodec,
ScanBodyInference/ScanGradientRules [preflight](../../../../glossary.md#term-preflight), [PrepareContext](../../../../glossary.md#prepare-context-preparecontext).validateScan and CpuGeneralScanLowering
own independent new-[topology](../../../../glossary.md#term-topology) rejection. Existing private helpers stay in current owners.
MAIN owns root/scoped/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/[API](../../../../glossary.md#term-application-programming-interface-api) guides/plans/changelog; implementer owns affected source/Javadocs/tests.

## Dependencies and integration

- Depends on: Model0045 and practical Compiler0018/Checkpoint0014/NN0033 Complete, CPU0078 integration,
  approved ADR0059 design, coordinated source/count/consumer authority, exact Ready verification.
- Conflicts with: Scan header/source/compiled body, typed consumers/guards, CPU GeneralScanLowering,
  raw codec/cold copies and affected tests/docs. No concurrent all-PURE/typed writer.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [codec glossary](../../../../glossary.md#term-codec); [PURE glossary](../../../../glossary.md#term-pure))</span>
- Parallel group: None; source-typing cutover serial after CPU0078. Engine0022 cleanup is unrelated.
- Common base revision: exact accepted dirty source/authority [manifest](../../../../glossary.md#term-manifest) after CPU0078, captured at [Ready](../../../../glossary.md#term-ready).
- Integration order: authority/Ready -> isolated complete source/callers/guards -> independent review ->
  MAIN source constructor/negative-consumer and old-wire/CPU compatibility gates -> typed consumer source.
- Integration validation: Model/Javadocs, affected Compiler/Planning/Prepare/CPU/Checkpoint/NN tests,
  direct raw guards and actual source-mode CLI; no native/timing campaign; exact selectors at Ready.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [CLI glossary](../../../../glossary.md#term-cli))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

## Acceptance

Both directions, T/B=0/1 and named B: emission-only all-valid/length source and length-mode zero-ROW
SHARED accumulators have exact counts, last-length actual, [canonical](../../../../glossary.md#term-canonical-form) ports, [shapes](../../../../glossary.md#shape)/flags/[aliases](../../../../glossary.md#aliasing-and-arena) and
no fake state. Invalid empty semantic roots, counts, roles, changed [carry](../../../../glossary.md#term-carry) type/shape or contradictory
[bindings](../../../../glossary.md#term-binding) fail before output [identities](../../../../glossary.md#term-identity-meanings). Full input/output arithmetic remains checked independently
of empty [axes](../../../../glossary.md#term-axis); no first-carry access or guessed [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters). No storage read/[runtime](../../../../glossary.md#term-runtime) state is introduced.

Retain old seven/nine-form semantics through migrated seven/ten forms, including [homogeneous](../../../../glossary.md#term-homogeneous) cold
v4/v5 byte equality and canonical [capture](../../../../glossary.md#term-capture-meanings)/reconstruction. Independently reject new topologies at
raw [Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/cold/wire seams, including crafted attrs/body and zero-domain cases. Remove
incidental reflection/member-name/diagnostic pins rather than repinning new shapes/counts.

The actual [CLI](../../../../glossary.md#term-cli) constructs source and observes explicit mode/count/port [metadata](../../../../glossary.md#term-metadata) plus truthful current
consumer rejection; this is source-surface proof, not execution completion. Permanent tests [target](../../../../glossary.md#term-target)
real [cardinality](../../../../glossary.md#term-cardinality), alias, metadata or [admission](../../../../glossary.md#term-admission) defects. No builds/tests/lint/formatters mid-flight;
MAIN runs the final source-matched gate after independent source/[Javadoc](../../../../glossary.md#term-javadoc) review.

## Result

MAIN integrated the exact33reviewed worker paths, the independently approved removal of an obsolete
[Compiler](../../../../glossary.md#term-compiler) reflection/copy/count test, and the sole current-ledger formatter in one35-file source
cutover. No nine-[argument](../../../../glossary.md#term-argument) compatibility [alias](../../../../glossary.md#aliasing-and-arena) or dummy [carry](../../../../glossary.md#term-carry) remains.

Actual Model1210, Compiler418, Planning73, Prepare99, CPU1474, Checkpoint80, NN566 and current
conformance78 passed;60existing [CPU](../../../../glossary.md#term-cpu) cases skipped. [Model](../../../../glossary.md#term-model)/Compiler/[Checkpoint](../../../../glossary.md#term-checkpoint) Javadocs and required
jars passed. The first full gate remains FAILED at the obsolete copied Scan minimum-input tuple;
the owning test was deleted rather than repinned. Its affected full Compiler after-gate passed.

The public [CLI](../../../../glossary.md#term-cli) passed96F32/[F64](../../../../glossary.md#floating-point-precision-and-rounding) emission-only/zero-ROW Shared cases across both directions,
T/B=0/1/named B, explicit mode, aliases, [canonical](../../../../glossary.md#term-canonical-form) ports and no source storage; all96current
[Engine](../../../../glossary.md#term-engine) consumers rejected these source-only topologies. The unchanged compatibility driver
captured14qualified families/28actual forward/reverse CPU [kernels](../../../../glossary.md#kernel) before and after: all85SHA
records, including canonical [IR](../../../../glossary.md#term-intermediate-representation-ir) fingerprints, both [identity](../../../../glossary.md#term-identity-meanings) streams and complete class bytes,
match. Actual current-ledger rendering remains the exact306rows; no new capability is inferred.

model0046-integrated-files.json, first-failed-gate, compiler-final, actual-cpu-compatibility and
current-ledger-[authentication](../../../../glossary.md#term-authentication) retain exact sources, commands/XML and limits. Independent final
evidence/docs reviews returned APPROVE0. Required all-[PURE](../../../../glossary.md#term-pure) composition and typed reverse remain open.

## Exact [Ready](../../../../glossary.md#term-ready) launch

MAIN froze HEAD 07a281edd3086efde2853211b71b48288db027e1 plus all 2,647 dirty source-file [hashes](../../../../glossary.md#term-hash)
in model0046-base.json and created model0046-worktree. The implementation owns the complete
serial source/callers/guards cutover; no concurrent typed/all-[PURE](../../../../glossary.md#term-pure) writer. The independent [Metal](../../../../glossary.md#term-metal)
[lifetime](../../../../glossary.md#term-lifetime) [slice](../../../../glossary.md#slice) is read-only and source-disjoint. All docs/resources and the shared
LowPrecisionCapabilityLedgerTest source remain MAIN-only, including its constructor/formatter
migration during the separately owned current Metal basis refresh. No child executable checks.
