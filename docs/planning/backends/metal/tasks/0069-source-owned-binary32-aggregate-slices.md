# Task 0069: Source-Owned Binary32 Aggregate-Floor Slices

## Status

Review needed. The authorized shared proof substrate plus Slice 1 are complete and independently
approved after remediation, with zero remaining code, numerical-evidence, or security P0/P1/P2.
Slices 2 and 3 are not authorized and require their preceding checkpoint plus fresh approval.

## Change class

Class C — the eventual work changes numerical capability, generated-gradient reachability, source-owned GPU code, private preparation identity, and validation-before-mutation behavior.

## Goal

Build the smallest task-local binary32 and contributor-tree proof substrate, then consider three independently
shippable `ACCELERATOR`/`FLOAT32` custom-route slices in strict serial order: `L1_NORM`, `SCATTER_ADD`, then
`VARIANCE`. A failed proof, source audit, or runtime gate leaves that row false and promises no later slice.

## Baseline audit and decisions

- Common base is clean `69c07e30e2ec90c91de604c396896ee258e851e5`. Task 0067's current 115-wire ledger
  is structural `101/14`, capability `83/32`, MPSGraph `75/35/5`, custom `70/45/0`, schema 15, ABI 5, thirteen exports, route wires `1..3`, identity 22.
- Model owns exact mapping, contributor membership, empty/point/divisor rules, binary32 RNE
  primitive sites with set-valued DAZ/FTZ, and any all-terms-once binary aggregate tree. L1 is
  `ABS` then sum; scatter-add includes base and every matching update occurrence; variance is mean,
  one subtraction and square per contributor, aggregate sum, then `N-correction` division.
- Compiler keeps the existing formulas: L1's cotangent is sign-selected upstream with zero at
  signed zero/NaN; variance is `g * 2 * (x-mean) / (N-correction)`; scatter-add returns upstream for
  base, no index cotangent, and `gather(upstream, indices, axis)` for updates. This task does not
  alter those formulas or first-order classifications.
- Task 0060's MPSGraph recipes prove only structural execution. Current direct `SCATTER_ADD` and
  `VARIANCE`, and composed `L1_NORM`, are opaque and cannot authorize production. Every selected
  occurrence therefore uses one fixed source-owned `CUSTOM_PROGRAM`; the retained MPSGraph states
  remain descriptive and are never fallback candidates.
- Task 0053 remains `CORROBORATED_NOT_CERTIFIED`; its EXP partitions, finite checks, and source
  correspondence do not transfer. Only its useful proof discipline is retained: total raw-word
  classification, unsigned bounds, integer ties-to-even lemmas, source inventory, pinned tools, and
  explicit proof-status manifests. No Task-0053 theorem is treated as aggregate authorization.
- Task 0061's accepted source/compiler-site certificate is the recipe for binding exact kernel text,
  safe compiler mode, site inventory, and artifacts. Task 0069 adds no general numerical framework.

## Exact occurrence domains

All slices require `ACCELERATOR`, `FLOAT32` data/result, fully static positive canonical-dense
layouts, 32-bit-bounded element/byte/dispatch arithmetic, distinct authenticated value-table
buffers, and no gradient-bearing role on the selected operation. A generated Gather may require
gradients, but its emitted Scatter-Add node itself must satisfy that no-gradient domain.
`STRICT_IEEE`, dynamic/empty/noncanonical/aliased occurrences and every unlisted rank, axis,
carrier, attribute, or gradient combination remain false.

### Slice 1 — `L1_NORM` wire 114

- Input rank is exactly one with extent `N>0`; `N`, the four-byte input span, and dispatch count
  must each fit unsigned 32-bit arithmetic. Axes are exactly `[0]`, never empty.
  `keepDimensions=false` produces canonical rank zero; `true` produces canonical `[1]`.
- One output thread clears each input sign bit for exact raw `ABS`. Its reduction tree has exactly
  the `N` ABS values as leaves and exactly `N-1` actual safe binary32 additions in increasing
  contributor order; it has no zero leaf. `N=1` is the direct ABS leaf with no add site. The result
  stores once, with no invented identity, pre-truncation, fusion, epsilon, or hidden reduction.
- Either input zero becomes `+0`; NaN remains NaN; infinity yields `+infinity` unless a NaN is
  present; finite results are nonnegative.
- Gradient-bearing L1 remains false, so the existing sign/zero/NaN cotangent graph is not newly
  reachable.

### Slice 2 — `SCATTER_ADD` wire 70

- Data/result rank is exactly one with length `D>0`; indices and updates are rank one with common
  length `U>0`; axis is exactly zero. Indices are `INT32` or `INT64`, are a materialized partition
  feed, and update Shape is exactly the rank-one Gather result. All data/update/result gradient
  flags are false.
- A complete CPU-side scan validates every `0 <= index < D` before any command encoding, graph-node
  dispatch, or output mutation. Duplicate targets are legal and are never uniqueness-filtered.
- One output thread owns each target: it raw-loads base exactly once, scans update ordinals in
  increasing order, applies one safe binary32 add for every matching occurrence, and stores once.
  An unaddressed cell is a raw base copy, preserving NaN payload/quietness, subnormal bits, and
  signed zero. Addressed NaN, infinity, subnormal, overflow, and signed-zero results come only from
  the proved set-valued site chain; no final canonicalization is allowed. There are no atomics or
  write races.
- The same admitted no-gradient node closes the existing first-order `GATHER` data cotangent only
  for canonical rank-one `FLOAT32` data, rank-one materialized `INT32`/`INT64` indices, axis zero,
  positive static extents, and canonical cotangent. Compiler-generated zero base plus upstream
  updates must be captured and proved end to end. Gradient-bearing `SCATTER_ADD`, its base/update
  cotangents, Gather-Elements/Gather-ND, and higher order remain false; no origin discriminator is
  added. Slice 2 atomically replaces the backend contract's current additive-scatter fail-closed statement with only this exact domain and generated Gather closure; it cannot ship beforehand.

### Slice 3 — `VARIANCE` wire 112

- Input rank is exactly one with extent `N=1`, axes exactly `[0]`, correction exactly zero, and no
  gradient flags. Both keep forms target one canonical scalar cell; every other domain stays false.
- Exactly one dispatched output thread owns that cell and stores once. With no payload classifier
  or special publication, it evaluates only `mean=DIV(x,+1)`, `d=SUB(x,mean)`, `q=MUL(d,d)`, the
  singleton square-sum leaf `q`, then `DIV(q,+1)`. Each name is one safe binary32 site; there is no
  aggregate add, clamp, reciprocal substitution, FMA, stable replacement, or final tolerance.
- The proof must derive NaN for NaN/infinity and exact `+0` for every finite input, including both
  zeros and subnormals under every DAZ/FTZ choice, from those sites alone.

## Proof and source gates

1. Add task-local `Task0069Binary32.lean` and `Task0069ReductionTree.lean`: total binary32 raw
   classes; exact typed-one facts; set-valued RNE/DAZ/FTZ primitive relations; leaf-labelled binary
   trees/multisets; and the theorem that each source left chain is a permitted all-terms-once tree.
   L1's leaves are exactly the ABS values, internal nodes correspond one-to-one with the `N-1`
   source additions, and its singleton tree is the direct ABS leaf. No `sorry`, axioms, native
   floating oracle, or general framework.
2. Add per-slice Lean domain theorems. L1 proves raw ABS, point, class, and contributor obligations.
   Scatter proves base-plus-filtered-occurrence membership per target, duplicates retained,
   unaddressed raw identity, and one writer. Singleton variance proves literal-site membership,
   special classes, one-thread ownership, and one store without preclassification.
3. Bind the exact header, native dispatcher/preflight, compiler flags, function names, loop/site
   inventory, generated library, and proof inputs in a hash manifest plus source/compiler-site
   certificate. Compile with `MTLMathModeSafe`; reject fast math, reassociation outside the written
   tree, contraction, hidden reduction, and unbound artifacts. Device regressions corroborate but
   never authorize an uncovered domain.
4. Java and native independently authenticate profile, kind, attributes, carrier, rank, Shape,
   canonical layout, gradient flags, exact counts/bytes, buffer distinctness, dispatch/writer
   geometry, and correction/index constraints before allocation or encoding. Whole-program scatter
   payload validation precedes every dispatch and mutation; invalid execution preserves sentinels.
5. Each slice lands as a separate identity/capability/source/proof/test/docs checkpoint and receives
   independent code, evidence, and security review. Failure leaves its row and every later serial
   slice false until a reviewed amendment; already shipped earlier slices remain valid.

## Identity and count ledger

| Accepted checkpoint | Structural | Capability | MPSGraph | Custom | Schema / ABI / exports | Identity |
|---|---:|---:|---:|---:|---|---:|
| Baseline | `101/14` | `83/32` | `75/35/5` | `70/45/0` | `15 / 5 / 13` | 22 |
| Slice 1 only | unchanged | `84/31` | unchanged | `71/44/0` | `16 / 5 / 13` | 23 |
| Slices 1–2 | unchanged | `85/30` | unchanged | `72/43/0` | `16 / 5 / 13` | 24 |
| Slices 1–3 | unchanged | `86/29` | unchanged | `73/42/0` | `16 / 5 / 13` | 25 |

Each checkpoint advances all six backend-local workload/policy/candidate/compatibility/route/codec
identities together and rejects the prior identity and all older values. Slice 1 cleanly advances
the image from schema 15 to schema 16 so the first formerly reserved header word carries the exact
existing numerical-profile wire; schema 15 and every other schema reject. Identity remains 23
because its compatibility and codec already bind the profile and its workload digest already binds
the schema version and complete encoded image, so no schema-15 decision can match a schema-16
workload. Operation, attribute, type, and route wires do not change. At each checkpoint every other
false row remains false.

## Non-goals

No Model/API/Compiler-formula change, new operation/attribute/type/route wire, ABI/export change,
broad aggregate helper, strict-profile capability, noncanonical or empty geometry, positive
variance correction, L2 norm, mean expansion, additive Scatter-ND/Elements, timing, autotuning,
fallback, retry, atomics, host repair, opaque-selector inference, source-unbound corpus proof,
approximate acceptance, or final output epsilon. The schema-16 profile cell is the sole image
grammar revision.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence).
- [Foundational module contract — Numerical profiles / Aggregate floor / Composite inheritance](../../../../architecture/contracts/foundational-modules.md#numerical-profiles).
- [Backend execution contract — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend).
- [Backend execution contract — Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity).

If current source or an applicable contract disagrees with this brief, stop, keep the occurrence
false, and submit a reviewed plan amendment rather than inventing architecture.

## Dependencies and integration

- Depends on: Task 0068 Complete at approved `1775081a`; Task 0067 current ledger; current Model and
  Compiler contracts; retained Tasks 0053/0060/0061 evidence only as bounded above.
- Conflicts with: every concurrent Metal capability/native/custom-source/catalog/route/identity/
  package/shared-document edit, Compiler indexing/reduction/autograd edit, and resumed Task 0053
  production work.
- Parallel group: None.
- Common base revision: `69c07e30e2ec90c91de604c396896ee258e851e5`.
- Integration order: proof substrate + Slice 1, then Slice 2, then Slice 3; no batched cutover.
- Integration validation: the slice's proof/certificate gates, native package, complete Metal,
  focused Compiler, Metal conformance/public Engine, architecture/Javadoc, then repository test/build.
- Shared-document integration owner: Task 0069 executor, serially.

## Files and symbols

- `backends/metal/evidence/0069/` — pinned task-local proof, manifests, certificates, and checkers.
- `backends/metal/.../MetalCapabilityProvider.java`, `MetalOperationRouteCatalog.java`, preparation
  identity owners and focused tests — exact occurrence admission, fixed custom route, identities.
- `native/metal-macos-arm64/src/synaptik_task0069_aggregate_kernels.h` and
  `synaptik_metal_foundation.m` — source-owned kernels, independent preflight, fixed dispatch.
- Metal/Compiler/conformance/integration tests and active architecture/package documentation — exact
  slice behavior, generated Gather closure, false boundaries, counts, identity rejection, and the
  Slice-2 atomic update to `backend-execution.md` additive-scatter and generated-gradient text.

## Acceptance and validation

For each independently approved slice: its Lean proof and source certificate pass; the exact route
executes through the public Engine on the packaged dylib; special classes, duplicate contributors,
raw-copy cells, correction/count bounds, one-writer/store-once, both keep forms where applicable,
generated Gather closure, false neighboring domains, alias/error-before-mutation, identity
rejection, and complete count partitions pass. Native builds use warnings as errors. Permanent
tests defend these observable boundaries, not source text or incidental plumbing.

Worker validation is the existing Task-0068 native/package matrix plus the new `0069` proof runner,
complete `:backends:metal:test`, focused Compiler indexing/reduction/autograd tests, Metal
conformance and public Engine integration, architecture/Javadoc, `./gradlew test`, `./gradlew build`,
and `git diff --check`. No benchmark or timing entry point runs.

## Documentation and review impact

The implementation must update Metal package/Javadoc, backend execution identity/count/domain text,
this task, master plan, roadmap, and any affected native package evidence. Class C requires separate
code, numerical-evidence, and security/determinism review at every slice checkpoint.

## Result

The authorized shared substrate and Slice 1 are implemented. Metal now admits only accelerator
FLOAT32 no-gradient `L1_NORM` for one canonical positive-static rank-one input, ordered multi-axis
`[0]`, and canonical scalar or retained `[1]` output. Wire 114 always selects the fixed
source-owned `CUSTOM_PROGRAM` under `CA_0069`; Java and native creation reject the opaque direct
MPSGraph route. Java/native validation independently enforces the exact domain, unsigned-32-bit
element/four-byte-span/dispatch bounds, and distinct buffers. One
thread raw-clears every contributor sign bit, initializes from ordinal zero, performs exactly
`N-1` safe binary32 additions in increasing ordinal order, and publishes one logical result;
`N=1` performs no addition. SCATTER_ADD and VARIANCE remain production-false.

The two Lean files prove total raw classes; exact binary32 `+1`; exact ABS; complete set-valued
DAZ/FTZ alternatives; an explicit finite nonnegative nearest-even relation in `2^-149` units with
overflow-midpoint and even-significand tie rules; NaN, positive-infinity, and nonnegative non-NaN
class behavior; the aggregate L1 nonnegative-or-NaN invariant; reachable-source NaN absorption with
signed NaN allowed and every non-NaN operand sign-clear; positive-infinity persistence without NaN;
labelled contributor multiplicity; the source left tree's `N-1` add nodes; direct singleton ABS;
and source-fold membership constrained by `Binary32RneContract` rather than an arbitrary binary
function. Both compile without `sorry` or axioms. The source certificate pins
the exact kernel/foundation/proof inputs. The required compiled-MSL audit extracts
the exact assembled runtime source and compiles it with `DEVELOPER_DIR` fixed to signed Xcode 27.0
build 27A266a, Metal 32023.921, macOS SDK 27.0, `metal3.2`, no-fast-math, warnings as errors, and
the explicit SDK isysroot. Pinned source/AIR/metallib SHA-256 values are
`070c910ff3be6853720274d670fb5a573c490ceec60d680eb323d9001c965c10`,
`fc6258b8627ec607434f02fe63eb77dd05957f9b5821e95b157e1149f2fe3189`, and
`ffa3c751fbab39acff50c06f6fda720e4e831a3d06657ee459921f50d11ba8f3`. AIR inspection finds
exactly one unflagged `fadd` in `l1_norm_f32_0069`, raw sign masks and the ordinal loop, no other
floating arithmetic, and only the final raw serialization path.

Native warnings-as-errors build, fixed-identifier signing, canonical package verification, Gradle
package ingestion/ZIP creation, fresh extraction, extracted-package verification, and byte
comparison passed. The packaged dylib is `484944` bytes with SHA-256
`c036c78259b0d30db14e96ddf7ae205ae92f99901d7afc0be192f46b59a0e529` and ad-hoc CDHash
`6b64ae9f178b4b12b4d0f53dfd525b37f9758a0e`; the local ZIP is `141328` bytes with SHA-256
`4bb2822ac35f3a70ac76dbca72bb1a628efc3f9eefbadb1e05080f9c883cc235`.

The initial implementation checkpoint passed complete Metal, Compiler, conformance, public Engine
Metal integration, architecture, Javadoc, repository test, and serialized full build, totaling
`3718` tests with zero failure/error. Initial independent security review approved with zero
P0/P1/P2. Initial code and numerical-evidence findings required the explicit nearest-even/class
contract, raw native malformed-image coverage, direct identity-22 rejection, and current ledger
documentation; cumulative review also identified and closed the forceable direct-MPSGraph L1 seam.
External design review then required the clean schema-16 profile cutover and narrowed the RNE NaN
premises to signed NaN or sign-clear non-NaN reachable operands; the generic source fold carries
the same invariant.

After that remediation, the proof/source-certificate/compiled-MSL runner, focused raw native schema
and L1 tests, identity tests, affine schema fixture, and direct public L1 Engine scenario all pass.
The rebuilt signed package, Gradle ingestion/ZIP, fresh extraction, extracted-package verifier, and
byte comparison pass. The final unfiltered serialized `./gradlew test --rerun-tasks
--max-workers=1` passes all 76 actionable tasks in 3m02s. The subsequent unfiltered serialized
`./gradlew build --rerun-tasks --max-workers=1` passes all 87 actionable tasks in 2m57s, including
the unchanged `CpuPartitionDagGeneratedEvidenceTest`; no timing policy, exclusion, suppression, or
waiver is used. Current ledgers are capability `84/31`, structural `101/14`, MPSGraph `75/35/5`,
custom `71/44/0`, and identity 23; schema 16 binds the exact supported profile wire, while ABI 5,
thirteen exports, and all other false rows are unchanged. Slice 2 remains outside this checkpoint.
