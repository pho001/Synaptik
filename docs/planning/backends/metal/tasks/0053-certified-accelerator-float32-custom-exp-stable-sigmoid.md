# Task 0053: Certified ACCELERATOR FLOAT32 Custom EXP and Stable SIGMOID

## Status

Blocked

This planning-only successor supersedes Task 0051 for all future `EXP`/`SIGMOID` realization work
without rewriting its historical evidence. No candidate is currently `DOMAIN-PASS`. The task stays
Blocked until concrete custom source, frozen constants, and a pinned machine-checkable
complete-domain proof package pass independent read-only review.

Only after that prerequisite passes may this task consume its one newly authorized real-device
numerical oracle. Task 0051's consumed invocation is never rerun, amended, or relabeled. If the
oracle leaves one survivor for an operation, no comparative cost gate is required. If it leaves
multiple survivors with identical proven semantics/domain, selection may close without timing only
through strict structural dominance; otherwise it remains pending an explicitly authorized
controlled environment. Local developer-device timing is diagnostic only and never qualification,
route-selection, or tuning-identity authority. This plan changes no capability, schema, identity,
route, production source, test, native binary, or package, and performs no device action.

## Change class and scope

Class C — eventual implementation would add canonical `FLOAT32` `EXP` and `SIGMOID` under
`NumericalProfile.ACCELERATOR`. This planning commit only defines proof, oracle, cost, selection,
and future implementation gates. It does not reserve wires or versions and does not authorize
staging through current production preparation.

In scope for future execution:

- one auditable custom `EXP` candidate over raw binary32 words;
- one stable custom `SIGMOID` candidate proved recursively from an exact guard and `NEG`, the proved
  `EXP`, and explicit `ADD`/`DIV` sites;
- any MPSGraph or composed alternative that independently earns complete-domain `DOMAIN-PASS`
  before candidate freeze; and
- one proof-first oracle, survivor adjudication under ADR 0022, one fixed-route decision only when
  authorized, then a separately reviewed production implementation.

Out of scope now and later unless a new task says otherwise: strict-profile `EXP`/`SIGMOID`, any
other unary kind, rank zero, dynamic or zero extents, views, non-`FLOAT32` payloads, gradients that
require unsupported topology, runtime checking, retries, fallback, input-dependent routing,
general transcendental frameworks, Model changes, Compiler rewrites, async/device-selection work,
native ABI exports, Task-0052 files/evidence, and reuse of a removed Task-0051 implementation.

## Supersession and preserved history

Task 0051 remains the immutable blocked record for its five historical candidates and sole 83-word
Apple M3 Max invocation:

- opaque direct MPSGraph `EXP` and its inherited stable composition passed that bounded sample but
  remain `DOMAIN-BLOCKED`;
- the original custom `EXP`, custom `SIGMOID`, and direct `SIGMOID` retain their recorded numerical
  failures; and
- its historical plan stopped at a trace prerequisite, but this successor does not inherit
  full-Xcode tracing as an unconditional blocker. Supported actual observation is needed only to
  establish structural facts for an opaque survivor.

Task 0053 owns only new source, new certificates, a new frozen candidate manifest, and one new
successor oracle after the prerequisites. It may reuse historical words as regression witnesses in
a materially new successor corpus, but it must not execute the Task-0051 source, binary, candidate
implementations, wrapper, or 83-word invocation again.

## Exact authorized occurrence domain

A future occurrence is eligible only when all of the following are true:

- profile is exactly `ACCELERATOR`;
- kind is parameterless `UnaryElementwiseKind.EXP` or `SIGMOID` with
  `NoOperationAttrs.INSTANCE`;
- there is exactly one input and one output;
- both descriptors are canonical dense-contiguous, zero-offset, non-view `FLOAT32`;
- input and output have the same fully static positive rank-`1..16` Shape and matching
  `requiresGrad == false`;
- element and byte geometry, launch products, offsets, and buffer bounds are checked before native
  resource creation; and
- the occurrence, topology, feeds, targets, ownership, state, and all sentinels/padding pass the
  then-current Java and native preflight before candidate creation.

Every strict occurrence and every excluded descriptor remains capability-false. Passing a bounded
rank or value sample never narrows or expands this domain.

## Exact EXP result-set obligation

Interpret an input word `x` as binary32. The complete proof must use the disjoint and exhaustive raw
partition below over all `2^32` words, with counts and coverage checked by the proof kernel:

1. positive and negative NaNs: exponent `0xff`, fraction nonzero;
2. positive and negative infinity: exponent `0xff`, fraction zero;
3. positive and negative zero: exponent and fraction zero;
4. positive and negative subnormals: exponent zero, fraction nonzero; and
5. positive and negative finite normals: exponent in `1..254`.

For a finite subnormal input, the Model input alternatives are the represented value and its
same-signed zero. For every other finite input, the represented value is the sole input alternative.
The candidate may choose one documented allowed DAZ path, but the checker must construct the full
alternative set and prove the chosen result belongs to it.

For each chosen finite input value, evaluate mathematical `exp` with a certified directed enclosure
and define `r` as its correctly rounded binary32 round-to-nearest-even result. The allowed result is:

- any binary32 NaN for a NaN input; payload, sign, signaling/quiet state are abstract;
- exactly positive zero for negative infinity;
- exactly positive infinity for positive infinity;
- exactly `0x3f800000` for either signed zero;
- for a finite, nonzero, non-subnormal `r`, only a finite nonnegative word whose inclusive ordered
  distance from `r` is at most `5`;
- for a nonzero subnormal `r`, exactly `r`, positive zero, or negative zero; and
- for an `r` that is zero or infinity because of correctly rounded underflow or overflow, the exact
  required class and sign, independent of the distance rule.

Ordered distance is mathematical absolute difference of the unsigned monotone keys

```text
key(bits) = sign_set(bits) ? unsigned(~bits) : unsigned(bits ^ 0x80000000)
distance(a,b) = abs_integer(key(a) - key(b)).
```

It is never a wrapping machine subtraction. NaN, zero, infinity, overflow, underflow, result sign,
and result class are checked before and independently from the ordinary finite distance. The final
ordinary exact-EXP bound is inclusive: `distance(actual,r) <= 5`.

## Stable SIGMOID recursive obligation

`SIGMOID` has no final elementary ULP/distance envelope. Its proof must follow the exact stable
site graph for every raw input and every Model alternative.

After explicit NaN classification, apply the exact raw sign guard. The selected branch is:

```text
negative-sign branch:
    e = EXP(x)
    d = ADD(1.0f, e)
    y = DIV(e, d)

nonnegative-sign branch:
    n = NEG(x)
    e = EXP(n)
    d = ADD(1.0f, e)
    y = DIV(1.0f, d)
```

`NEG` is exact sign-bit inversion with its special-value behavior proved. `EXP` is exactly a
`DOMAIN-PASS` instance of the custom candidate proved above, not a selector with analogous output.
At each `ADD` and `DIV` site, the proof independently enumerates same-signed-zero DAZ alternatives
for subnormal operands, performs exactly one binary32 round-to-nearest-even operation, and admits an
exact nonzero subnormal result or either signed zero as that site's FTZ alternatives. NaN-producing
or NaN-consuming sites use the Model's abstract NaN class.

The proof must show exact branch selection, exact contributor/operand identity, and explicit
binary32 materialization between sites. A fused implementation may pass only if it proves the same
observable site boundaries and allowed rounding set; an opaque fusion assertion fails closed. The
final `SIGMOID` word must be a member of the recursively constructed site-by-site union. Agreement
with a high-precision whole-sigmoid value or final tolerance envelope is not evidence.

## Candidate ledger and freeze rule

| Candidate class | Current Gate 1B status | Future admission rule |
|---|---|---|
| `CUSTOM_EXP_CERTIFIED_V1` | `DOMAIN-PENDING` | exact source/constants/site manifest plus complete checked proof and independent approval |
| `CUSTOM_SIGMOID_STABLE_CERTIFIED_V1` | `DOMAIN-PENDING` | recursive proof using the exact approved EXP instance and proved guard/NEG/ADD/DIV sites |
| current opaque direct MPSGraph EXP | `DOMAIN-BLOCKED` | remains blocked; selector name, SDK declaration, and samples cannot admit it |
| current opaque direct MPSGraph SIGMOID | `DOMAIN-BLOCKED` | remains blocked and is not a stable-formula proof |
| Task-0051 MPSGraph stable composition | `DOMAIN-BLOCKED` | inherits its opaque EXP dependency and remains historical only |
| future MPSGraph or composed alternative | absent | may be added before freeze only with its own independently checked complete-domain proof for every primitive, shape, site, materialization, and deployment version |

No route family is a baseline. Before the oracle, commit one immutable candidate manifest naming
every considered route, operation, source/proof identity, and verdict. A candidate without
`DOMAIN-PASS` is recorded but is not compiled or run. Adding, removing, or changing source,
constants, compiler flags, proof assumptions, or a candidate after freeze invalidates the oracle
authorization and requires a newly reviewed successor plan; it never permits a retry.

## Gate 0: prerequisite before any source execution

The repository must contain the concrete candidate source/constants and all artifacts in the proof
package below; a pinned offline proof command must succeed; a second implementation-independent
all-raw-word checker must succeed; and independent read-only review must issue `DOMAIN-PASS` for
each candidate.

Trace tooling is not an unconditional prerequisite. Exact source-bound dispatch and route-owned
temporary-byte declarations are admissible for closed custom routes under ADR 0022. An opaque
survivor still requires supported observation of actual framework-internal dispatches and peak
transient resources before it can participate in structural comparison. If multiple nondominated
survivors remain after exact structural facts, a separately authorized controlled comparative
environment and protocol are required. A local developer timing run cannot satisfy that gate.

Until the proof prerequisite passes, Task 0053 remains Blocked and performs no device process,
compilation for execution, timing, capture, production edit, schema reservation, or capability
action.

## Gate 1A: concrete source and structural audit

The custom candidates must be fixed typed kernels, not a general transcendental subsystem. Before
Gate 1B review, retain reviewable source that:

- classifies raw words before arithmetic and contains explicit NaN/infinity/zero/subnormal paths;
- calls no Metal `exp`, `exp2`, `sigmoid`, fast transcendental, MPSGraph selector, or other opaque
  numerical routine inside the custom proof boundary;
- freezes range-reduction representation, table dimensions, exact table bytes, polynomial or other
  approximation, coefficient bytes, evaluation order, exception records, and output packer;
- uses only operations enumerated in the source model and has checked shifts, carries, conversions,
  indices, offsets, and buffer lengths;
- compiles with safe math, fast/relaxed math disabled, and an explicit contraction policy;
- contains no undefined signed overflow, implementation-defined conversion, unchecked table read,
  hidden allocation, or data-dependent fallback; and
- preserves exact input/output Shape and one-output pointwise ownership for every admitted rank.

The artifact manifest must enumerate every binary32 `ADD`, `SUB`, `MUL`, `DIV`, conversion, and
`FMA` site by stable source identifier. For each site it records operand interpretation, DAZ choice,
rounding mode, contraction, intermediate materialization, FTZ choice, and exceptional-class rule.
An integer-only section records an explicit empty hardware-FLOAT32/FMA set and proves from source and
compiler inspection that the set is truly empty. An unlisted compiler-created FMA, contraction,
conversion, or intrinsic rejects the candidate.

## Gate 1B: machine-checkable complete-domain proof

Gate 1B precedes and is independent of the numerical oracle. The proof must cover every raw input
and every authorized Shape, not a sample or an error estimate copied from a generator log.

### Mandatory EXP theorems

The checked package proves:

1. raw-class partition disjointness, exact counts, and union equal to all `2^32` words;
2. NaN/infinity/zero/subnormal behavior and the represented/same-signed-zero DAZ alternatives;
3. exact conversion from binary32 to the candidate's integer/fixed-point or proved floating
   representation, including sign, shift, carry, and width bounds;
4. unique range-reduction quotient selection and a directed enclosure for the residual, without
   treating an irrational constant such as `ln(2)` as exact;
5. every table-index formula is total and in bounds, every table interval encloses its mathematical
   value, and every scaling exponent/shift is in range;
6. the polynomial or other approximation is valid over the entire certified residual interval,
   with approximation, coefficient-rounding, evaluation, truncation, and reconstruction errors
   composed in the correct direction;
7. every actual binary32 RNE/FMA site has the exact declared one-rounding semantics, and every
   possible DAZ/FTZ alternative is represented at the corresponding Model site;
8. exact adjacent-word overflow and underflow thresholds, gradual-subnormal packing, tie-to-even
   boundaries, and infinity/zero publication;
9. every hard-to-round/exception table is sorted, unique, in bounds, and exhaustively covers all
   unresolved intervals; any unresolved word rejects the proof; and
10. for every finite allowed input path, the final result has the required class/sign and either the
    exact subnormal/zero/infinity membership or inclusive ordinary ordered distance `<=5`.

### Mandatory launch and memory theorems

For rank `r in 1..16`, positive dimensions `d[0..r-1]`, and
`N = product(d[i])`, prove checked element and byte products and the launch mapping

```text
linear(x,y,z) = x + W * (y + H * z)
```

for the exact host grid construction. The proof must show products fit their types, every integer
`0 <= linear < N` has exactly one grid coordinate, excess grid threads write nothing, each admitted
thread reads and writes exactly one in-bounds element, and no 32-bit truncation occurs. The theorem
must cover every resource-feasible positive rank-`1..16` Shape, not only the future rank witnesses.

### Mandatory SIGMOID theorems

The checked package proves:

1. total NaN handling and the exact raw sign guard, including both signed zeros and infinities;
2. `NEG` source-bit behavior for every non-NaN word;
3. recursive use of the exact approved EXP theorem for the selected branch;
4. each `ADD`/`DIV` implementation against complete raw-word partitions, including exceptional
   classes, DAZ, one RNE operation, FTZ, signed zeros, and materialized intermediates;
5. no contributor/site is omitted, duplicated, reordered, fused without proof, or replaced by a
   whole-function approximation; and
6. for every one of the `2^32` inputs, the published word belongs to the Model's recursively
   enumerated branch result set.

### Retained proof artifacts

Before the oracle, retain under a dedicated Task-0053 evidence directory, with SHA-256 identities:

- exact `.metal` source and host staging source;
- machine-readable candidate, raw-partition, FLOAT32-site, constants, table, polynomial, exception,
  launch, and resource manifests;
- constant/coefficient generator inputs and deterministic generated outputs;
- the formal source model and theorem sources;
- proof-kernel outputs/certificates and a single offline proof driver;
- an independent exhaustive `2^32`-word checker, its source, exact-reference configuration, partition
  counts, class counts, maximum ordered distance and witness, unresolved count, exception count,
  and deterministic report;
- compiler version/options, source-to-model/site correspondence audit, and compiled-function
  reflection or equivalent instruction/site audit; and
- one top-level manifest binding every byte, tool version, command, assumption, and result.

Evidence remains reviewable after execution. Only compiled binaries, raw temporary caches, and
unneeded transient workspaces are disposable.

### Acceptable proof tools

The numerical and bit-level theorems must terminate in a small proof kernel. Accepted primary paths
are either:

- a pinned **Lean 4** toolchain checking the raw-word, IEEE/RNE, interval, approximation, indexing,
  and launch theorems; or
- pinned **Coq + Flocq**, with **Gappa** allowed to generate interval/rounding certificates that
  `coqc` checks.

**Sollya** may generate approximation coefficients and exact rational interval obligations;
**MPFR/MPFI** with directed rounding may drive the independent adaptive-precision reference and
all-word cross-check. They are generators/checkers, not substitutes for a proof-kernel result.
An SMT solver, CBMC/Kani run, compiler optimization report, Python/decimal script, random test,
spreadsheet, ULP sample, or `PASS` log is corroboration only unless every relied-on result is emitted
as a certificate replayed by the accepted Lean/Coq kernel.

The future implementation must pin exact tool/library versions and distribution hashes before use;
this plan intentionally does not guess versions or certificate bytes.

### Fail-closed independent review

The independent reviewer is read-only and reruns the proof from a clean, network-disabled
environment. The review rejects the candidate on any:

- missing or hash-mismatched artifact, mutable download, unpinned tool/library, nondeterministic
  generator, or unreproducible report;
- `sorry`, `admit`, candidate-specific axiom, unchecked foreign callback, proof timeout, solver
  `unknown`, or unreviewed trusted assumption;
- raw-word partition gap/overlap/count mismatch, unproved DAZ/FTZ branch, unresolved interval,
  ambiguous rounding boundary, or maximum distance greater than `5`;
- range-reduction gap, invalid constant enclosure, table/index/shift/carry/buffer bound gap,
  duplicate exception key, or incomplete exception proof;
- source/model/constants/compiler-options mismatch, unmodeled FLOAT32/FMA/contraction site, opaque
  numerical call, or compiler-created operation not present in the site manifest;
- SIGMOID final-envelope argument in place of recursive site membership; or
- launch proof that omits a rank, positive checked extent, resource-feasible element count, exact
  one-writer coverage, or in-bounds access.

Any rejection assigns that candidate `DOMAIN-BLOCKED`, records the first failure, and forbids the
device oracle. There is no guessed certificate, reviewer waiver, post-oracle proof repair, narrowed
domain, or conditional `DOMAIN-PASS`.

## Gate 2: one new proof-first numerical oracle

This task explicitly authorizes exactly one future real-device numerical process invocation, but
only after Gate 0, Gate 1A, Gate 1B, independent approval, and immutable candidate/corpus manifests
all pass. A failed process, crash, mismatch, or control failure is recorded as the result; there is
no retry, second optimization level, second device/context, amended corpus, or rerun.

The successor corpus is generated and hash-frozen before execution from:

- fixed representatives of every raw class/sign, both zeros/infinities, quiet/signaling NaNs, and
  minimum/maximum subnormals/normals;
- Task-0051 failure and boundary words as historical regression witnesses, without running its
  source or reproducing its old 83-word-only invocation;
- every proved overflow/underflow and normal/subnormal threshold neighbor required by the manifests;
- predecessor/exact/successor words at every range-reduction quotient and table-index transition;
- binary32 words nearest each approximation-interval endpoint and every certified critical/error
  point;
- deterministic stratified exponent/fraction/sign samples bound to a published seed and generator
  hash; and
- positive canonical rank witnesses `1..16`, with checked Shapes and identical per-candidate values.

The exact generated word list, Shapes, order, generator, and SHA-256 are reviewed and frozen before
launch. The single process uses one default-device context, compiles every `DOMAIN-PASS` candidate
before creating the command buffer, disables relaxed/fast math, records compilation settings, uses
one command buffer where structurally possible, supplies direct targets, snapshots inputs, guards
outputs, and prints raw words plus control metadata. Every candidate sees the same ordered values and
Shapes.

An independent offline validator uses adaptive directed MPFR/MPFI enclosures for EXP and the formal
recursive site model for SIGMOID. It records per candidate: source/proof/corpus/output identities,
class/sign failures, maximum ordinary ordered distance and witness, DAZ/FTZ observations, recursive
membership, every rank/control verdict, and first failure. The bounded oracle is regression evidence
only; it can reject but cannot establish `DOMAIN-PASS`.

## Gate 3: survivor adjudication and conditional controlled comparison

Only candidates with Gate-1 `DOMAIN-PASS` and Gate-2 numerical `PASS` survive. For each operation:

1. zero survivors keeps capability false;
2. one survivor needs no comparative cost gate;
3. multiple survivors may be selected without timing only when they have identical proven
   semantics/domain and one strictly structurally dominates every alternative: no more compute
   dispatches, no more route-owned temporary bytes, and at least one strict improvement; and
4. multiple nondominated survivors leave route selection pending an explicitly authorized,
   controlled comparative environment and protocol.

Closed custom routes may establish dispatch and route-owned temporary-byte facts from exact
source-bound declarations under ADR 0022. Opaque routes require supported actual-internal
observation; backend calls, graph nodes, command buffers, inferred fusion, estimates, and process
RSS are not substitutes. Structural comparison is not a weighted score and cannot trade more
dispatches for fewer bytes. Timing on an uncontrolled local developer device may be retained as
diagnostic history only and never qualifies a route, selects a route, changes tuning identity,
populates a production decision, or breaks a structural tie.

## Gate 4: fixed route and future production cutover

Encode exactly one immutable production route only after Gate 3 authorizes it, with no threshold,
retry, fallback, runtime timing, or input-dependent choice. A route decision still does not
authorize production by itself. A future implementation revision must update all callers
atomically, prove schema/identity/preflight/lifecycle behavior, run focused consumer-observable
verification, and pass independent Class C review.

Task 0055 has landed schema 13, `EXP=55`, `SIGMOID=64`, and backend-local identity version 14 as
structural vocabulary only. Task 0053 must consume those stable values without advancing them
unless its production change introduces a genuinely new identity fact. No dual decoder, alias,
migration shim, or stale-byte acceptance is permitted.

## Future production and lifecycle proof

A selected implementation must preserve whole-partition analysis, canonical state, stable ordered
nodes/values/feeds/targets, declarations-before-assignment, transactional finalization, immutable
splats, direct assigned targets, context child leases, run-owned outputs/address workspace,
synchronous invocation, reuse, concurrent/session isolation, rollback, close rejection,
publication, transfer, trace, and tuning behavior.

Focused proof must cover capability positives/negatives, Java/native schema agreement, stale
identity rejection, all rank/geometry boundaries, mixed partitions with existing canonical
operations, repeated/concurrent/session isolation, and real public Engine forward execution without
CPU owner or skip. Generated `EXP` and `SIGMOID` gradients remain outside capability unless their
actual compiler topology is already closed by supported canonical operations; do not handwrite a
substitute or widen binary/affine/scalar capability.

## Dependencies and integration

- Depends on: completed and independently approved Model 0030/0031; Task 0051's preserved blocked
  record; Complete Task 0055's ABI-5/schema-13 foundation; Compiler unary inference/gradient
  capture; Config 0006; and Engine 0018.
- Planning base: `1f5c715c0e727d8bd56051f027c44cbed5991f81`.
- External blocker: the concrete proof prerequisite defined in Gate 0. A controlled environment is
  a later conditional blocker only if multiple nondominated survivors remain.
- Supersedes: Task 0051 for all future EXP/SIGMOID candidate, proof, oracle, cost, route, schema, and
  production work; it does not alter Task 0051's status or evidence.
- Independent of: Task 0052 candidates, proof, oracle, raw timing history, and production kernels;
  it shares only ADR 0022's corrected decision-authority rule.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.
- Parallel group: none.
- Integration owner: future Task-0053 implementer after the proof blocker resolves.

## Expected future files and artifacts

The proof-first revision is expected to add a dedicated retained Task-0053 evidence directory with
the exact source/manifests/certificates described above, without touching Task-0052 evidence. A
later selected-route implementation may update:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`,
  `MetalNegPartitionPreparer`, and affected package Javadocs;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, route decisions,
  and complete-plan compatibility tests;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and its README;
- focused capability/schema/native/prepared/candidate/codec/conformance/Engine tests; and
- Metal/backend/API/glossary explanatory docs, this task, the Metal master, and roadmap.

This planning commit creates only this task and updates the Metal master/roadmap frontier.

## Acceptance criteria

1. Task 0053 is the sole future EXP/SIGMOID successor; Task 0051 remains an unchanged historical
   blocked record and its oracle is never rerun.
2. Status remains Blocked until a concrete independently approved proof package/toolchain exists.
3. Gate 1B requires concrete source/constants and kernel-checked proof over the complete raw-word
   partition, DAZ, overflow/underflow, range reduction, table/index bounds, approximation interval,
   every RNE/FMA site, FTZ, inclusive EXP distance `<=5`, and rank-`1..16` launch/index coverage.
4. Stable SIGMOID is proved recursively from its exact guard/NEG, the approved EXP, and explicit
   ADD/DIV sites with intermediate materialization; no final envelope is accepted.
5. The candidate set contains the certified custom routes plus only independently domain-proved
   MPSGraph/composed alternatives. Every current opaque direct/inherited route stays blocked.
6. Fail-closed artifact/tool/review rules reject missing, guessed, partial, sampled, unpinned,
   source-mismatched, unresolved, or opaque proof claims before any device action.
7. After proof approval, exactly one new Task-0053 oracle may run over the frozen successor
   candidates/corpus; a failure is final and Task 0051 is not rerun.
8. Gate 3 closes a single-survivor row without comparison, permits timing-free multi-survivor
   selection only by strict structural dominance under identical proven semantics/domain, and
   otherwise leaves selection pending a controlled environment. Local timing is diagnostic only.
9. This planning change performs no build, test, native compile, device run, timing, capture,
   capability/schema/identity reservation, production edit, or Task-0052 evidence edit.
10. Task/master/roadmap links, fences, final newlines, statuses, path scope, and `git diff --check`
    pass, and the planning-only change is committed for independent review.

## Planning validation

Validate documentation only:

```bash
git diff --check
```

Also check Markdown links, anchors, fences, final newlines, status consistency, exactly the new task
plus Metal master and roadmap planning files, and zero production/native/test/evidence/generated
artifacts. Do not run a build, test, compiler, proof tool, native process, device oracle, timing
workload, or trace capture.

## Documentation and review impact

The Metal master and roadmap name Task 0053 as the blocked successor frontier, retain Task 0051 as
consumed historical evidence, preserve Complete Task 0052's raw timing only as diagnostic history,
and state that no Metal task is Ready. Independent planning review checks only the completeness and
fail-closed character of this plan. A later proof review is a separate read-only review over exact
retained bytes and proof-kernel results.

## Result

Planning-only Blocked successor created. No candidate source, constants, certificate, oracle,
measurement, route, capability, or production change exists yet. Task 0055 has since supplied its
schema-13 structural vocabulary. The next action is not a device run: it is a separately committed
concrete source/proof package plus pinned offline verifier, followed by independent read-only
`DOMAIN-PASS` review.
