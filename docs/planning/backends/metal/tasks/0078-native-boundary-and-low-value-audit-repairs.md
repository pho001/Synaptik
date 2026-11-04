# [Metal](../../../../glossary.md#term-metal) 0078 — Repair native boundaries and exact low-value behavior

## Reader context

These repair different boundaries: WHERE needs complete per-role shape metadata; untouched ScatterAdd cells need raw copying; scan initialization uses the actual first contributor; native buffers need their exact creating context. None relaxes numerical meaning or adds an operation. Same physical GPU does not mean same native owner.

Run calls borrow authenticated inputs and write assigned outputs only after validation. A context identity token must survive comparison without retaining or authenticating a reused context address. Historical totals/source receipts remain those of this repair; the current resource explanation is separate.

- [Current exact creating-context lifetime](../../../../backend-guide/metal-backend.md#completion-and-device-topology) — Explains same-device versus same-context ownership.
- [Current failure-before-write authentication](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Connects metadata and raw-copy repairs to native validation.

## Status

Complete

## Goal and class

Class C native/[ownership](../../../../glossary.md#term-ownership-meanings)/numerical correctness repairs for project-review D1–D4. Preserve
existing capability and exact [Model](../../../../glossary.md#term-model) contracts; no new [operation](../../../../glossary.md#operation) or route [qualification](../../../../glossary.md#term-qualification) program.

## Confirmed source findings and [scope](../../../../glossary.md#term-scope)

1. [Rank](../../../../glossary.md#term-rank)-16 WHERE false-branch [shape](../../../../glossary.md#shape) packs rank plus 16 [dimensions](../../../../glossary.md#dimension) into `inputExtents[16]`.
   Replace that host/MSL packing with the existing full role-rank/dimension metadata consistently
   for exact and low WHERE. Test false branch rank16 and last-axis broadcasting, not only the
   already tested condition/true/output roles.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [metadata glossary](../../../../glossary.md#term-metadata); [axis glossary](../../../../glossary.md#term-axis); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
2. `lp_scatter_add` widens/narrows unaddressed base words and canonicalizes their NaNs. Preserve
   raw unaddressed BF16/F16 words; apply arithmetic/narrowing only to actually addressed cells.<span data-glossary-navigation="true"> ([BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>
3. `lp_scan_body` adds the first sum contributor to an artificial +0. Seed from the first
   contributor as required; retain exclusive empty-prefix identities and exact zero-sign rules.<span data-glossary-navigation="true"> ([Seed glossary](../../../../glossary.md#term-seed))</span>
4. Native buffer wrappers carry device but no exact creating-context identity. Enforce the
   existing exact-context contract in custom-program, MPSGraph and singleton-NEG invocation;
   same MTLDevice is insufficient. Choose ownership-safe identity with no dangling/reused pointer
   acceptance, retain existing release lifetime, and reject before target mutation.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [device glossary](../../../../glossary.md#term-device); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [singleton glossary](../../../../glossary.md#term-singleton); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [release glossary](../../../../glossary.md#term-release); [lifetime glossary](../../../../glossary.md#term-lifetime); [target glossary](../../../../glossary.md#term-target))</span>

Own native [Metal](../../../../glossary.md#term-metal) foundation/low header and smallest associated Java [identity](../../../../glossary.md#term-identity-meanings)/tests, [conformance](../../../../glossary.md#term-conformance),
source/package evidence and focused documentation changes. No [Engine](../../../../glossary.md#term-engine)/[Compiler](../../../../glossary.md#term-compiler)/Training edits;
Engine 0019 separately restores [heterogeneous](../../../../glossary.md#term-heterogeneous) fixtures. ABI7/schema20 stay unless actual format
or export changes require a justified version change. Advance relevant semantic [compatibility
identity](../../../../glossary.md#identity-meanings-and-algebraic-identities), update its owning scoped version text and focused explanation atomically, and rebind
active source-[authentication](../../../../glossary.md#term-authentication) evidence. Do not rewrite historical proofs or frozen v1 ledger.

## Required contracts

Read AGENTS; root [scope](../../../../glossary.md#term-scope) index/Core invariants/[ownership](../../../../glossary.md#term-ownership-meanings)/dependencies/Testing; [backend](../../../../glossary.md#backend)-execution
headings [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module), Low-[precision](../../../../glossary.md#term-precision) capability evidence, [Metal](../../../../glossary.md#term-metal) backend, Profile-free
[backend capability](../../../../glossary.md#backend-capability) and [identity](../../../../glossary.md#term-identity-meanings); relevant [Model](../../../../glossary.md#term-model) WHERE/scatter/[scan](../../../../glossary.md#term-cumulative-scan) and numerical family contracts;
[runtime](../../../../glossary.md#term-runtime)-[prepare](../../../../glossary.md#prepare)-[engine](../../../../glossary.md#term-engine) persistent-resource/Prepare/[Run](../../../../glossary.md#term-run) [lifetime](../../../../glossary.md#term-lifetime) headings as used. Inspect native
[metadata](../../../../glossary.md#term-metadata)/three run paths/[release](../../../../glossary.md#term-release) and focused tests, native README/build/package [validation](../../../../glossary.md#term-validation).

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Rank16 WHERE false branches work for F32/F16, including broadcast variants; no metadata
  out-of-bounds read/write remains in either affected kernel.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [metadata glossary](../../../../glossary.md#term-metadata); [kernel glossary](../../../../glossary.md#kernel))</span>
- Low ScatterAdd preserves untouched signed/payload NaNs, signed zeros, and subnormal words for
  both low dtypes and both index widths; addressed values retain arithmetic semantics.<span data-glossary-navigation="true"> ([subnormal glossary](../../../../glossary.md#term-subnormal); [dtypes glossary](../../../../glossary.md#data-type--datatype))</span>
- Low scans cover forward/reverse × inclusive/exclusive, negative-zero first prefixes and
  empty-prefix identities, with independent raw-bit expectations.
- Direct native tests use two same-device contexts and reject foreign inputs/outputs for all
  three run paths before mutation; ordinary same-context execution and release remain valid.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [run glossary](../../../../glossary.md#term-run); [release glossary](../../../../glossary.md#term-release))</span>
- Native build/package verification; focused new tests, then one Metal module test/Javadoc and
  affected conformance run. Actual source-matched Apple-GPU execution is required evidence:
  try the authorized native-enabled environment if sandbox yields NO_DEVICE, and report any
  environmental block as a block, never a pass. Reuse existing active source/AIR workflow only
  where source hashes require renewal. No timing or broad historical oracle matrix.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [GPU glossary](../../../../glossary.md#term-gpu); [AIR glossary](../../../../glossary.md#term-air); [oracle glossary](../../../../glossary.md#term-test-oracle); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Independent Class C review checks algorithms, native identity lifetime, compatibility,
  source/package binding and docs. Coordinator owns final full integration checkpoint.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Commit](../../../../glossary.md#term-commit) scoped files, no push/shared plans; report commit/files/tests/doc impact and status.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0077 and [Model](../../../../glossary.md#term-model) 0032 Complete; source-confirmed project-review D1–D4.
- Conflicts with: other Metal/native/[schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings)/source-evidence/contract version writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS (other [backend](../../../../glossary.md#backend)/[module](../../../../glossary.md#term-module) [owners](../../../../glossary.md#term-owner) only).
- Common base revision: production `728680ea` plus [planning](../../../../glossary.md#term-planning)-only repair briefs.
- Integration order: standalone native repair; Engine 0019 final device validation uses this
  source-matched library after integration. No other worker edits Metal identity contract text.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [device glossary](../../../../glossary.md#term-device); [validation glossary](../../../../glossary.md#term-validation))</span>
- Integration validation: native/package/focused device/conformance evidence, independent review,
  combined repository and mixed-owner checkpoint.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [owner glossary](../../../../glossary.md#term-owner))</span>
- Shared-document integration owner: coordinator; worker owns only exact Metal version fact
  in owning contract and affected native/backend explanation/evidence.

Launch verified 2026-10-13: current root/scoped contracts explicitly require all four behaviors;
source confirms mismatches, predecessors Complete, no Metal writer. User authorizes bounded
repairs. Use clean context and isolated worktree, not new numerical architecture.

## Integration record

Implementation `50f570d8` plus independent-review documentation correction `12babf46`
integrated as `6e8d6b26` / `1db53756`. Final Class C review: APPROVE, no unresolved 0078 findings.
Four new actual-[device](../../../../glossary.md#term-device) tests passed; full [Metal](../../../../glossary.md#term-metal) 340 passed / 10 optional [observer](../../../../glossary.md#term-observer) skips,
37 [conformance](../../../../glossary.md#term-conformance) passed. Native build/sign/package/ZIP, active source/[AIR](../../../../glossary.md#term-air) [binding](../../../../glossary.md#term-binding), [Javadoc](../../../../glossary.md#term-javadoc) and
diff checks passed. Exact commands, environment and source [hashes](../../../../glossary.md#term-hash):
`backends/metal/evidence/0078/validation.json`. Coordinator also corrected the glossary corpus
fact. Shared version documentation was re-reviewed after correction.
The repair group's final combined checkpoint passed. D5 was separately closed by serial 0079.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
