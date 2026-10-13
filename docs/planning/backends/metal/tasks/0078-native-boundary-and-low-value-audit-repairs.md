# Metal 0078 — Repair native boundaries and exact low-value behavior

## Status

Ready

## Goal and class

Class C native/ownership/numerical correctness repairs for project-review D1–D4. Preserve
existing capability and exact Model contracts; no new operation or route qualification program.

## Confirmed source findings and scope

1. Rank-16 WHERE false-branch shape packs rank plus 16 dimensions into `inputExtents[16]`.
   Replace that host/MSL packing with the existing full role-rank/dimension metadata consistently
   for exact and low WHERE. Test false branch rank16 and last-axis broadcasting, not only the
   already tested condition/true/output roles.
2. `lp_scatter_add` widens/narrows unaddressed base words and canonicalizes their NaNs. Preserve
   raw unaddressed BF16/F16 words; apply arithmetic/narrowing only to actually addressed cells.
3. `lp_scan_body` adds the first sum contributor to an artificial +0. Seed from the first
   contributor as required; retain exclusive empty-prefix identities and exact zero-sign rules.
4. Native buffer wrappers carry device but no exact creating-context identity. Enforce the
   existing exact-context contract in custom-program, MPSGraph and singleton-NEG invocation;
   same MTLDevice is insufficient. Choose ownership-safe identity with no dangling/reused pointer
   acceptance, retain existing release lifetime, and reject before target mutation.

Own native Metal foundation/low header and smallest associated Java identity/tests, conformance,
source/package evidence and focused documentation changes. No Engine/Compiler/Training edits;
Engine 0019 separately restores heterogeneous fixtures. ABI7/schema20 stay unless actual format
or export changes require a justified version change. Advance relevant semantic compatibility
identity, update its owning scoped version text and focused explanation atomically, and rebind
active source-authentication evidence. Do not rewrite historical proofs or frozen v1 ledger.

## Required contracts

Read AGENTS; root scope index/Core invariants/ownership/dependencies/Testing; backend-execution
headings Concrete backend modules, Low-precision capability evidence, Metal backend, Profile-free
backend capability and identity; relevant Model WHERE/scatter/scan and numerical family contracts;
runtime-prepare-engine persistent-resource/Prepare/Run lifetime headings as used. Inspect native
metadata/three run paths/release and focused tests, native README/build/package validation.

## Acceptance and validation

- Rank16 WHERE false branches work for F32/F16, including broadcast variants; no metadata
  out-of-bounds read/write remains in either affected kernel.
- Low ScatterAdd preserves untouched signed/payload NaNs, signed zeros, and subnormal words for
  both low dtypes and both index widths; addressed values retain arithmetic semantics.
- Low scans cover forward/reverse × inclusive/exclusive, negative-zero first prefixes and
  empty-prefix identities, with independent raw-bit expectations.
- Direct native tests use two same-device contexts and reject foreign inputs/outputs for all
  three run paths before mutation; ordinary same-context execution and release remain valid.
- Native build/package verification; focused new tests, then one Metal module test/Javadoc and
  affected conformance run. Actual source-matched Apple-GPU execution is required evidence:
  try the authorized native-enabled environment if sandbox yields NO_DEVICE, and report any
  environmental block as a block, never a pass. Reuse existing active source/AIR workflow only
  where source hashes require renewal. No timing or broad historical oracle matrix.
- Independent Class C review checks algorithms, native identity lifetime, compatibility,
  source/package binding and docs. Coordinator owns final full integration checkpoint.
- Commit scoped files, no push/shared plans; report commit/files/tests/doc impact and status.

## Dependencies and integration

- Depends on: Metal 0077 and Model 0032 Complete; source-confirmed project-review D1–D4.
- Conflicts with: other Metal/native/schema/identity/source-evidence/contract version writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS (other backend/module owners only).
- Common base revision: production `728680ea` plus planning-only repair briefs.
- Integration order: standalone native repair; Engine 0019 final device validation uses this
  source-matched library after integration. No other worker edits Metal identity contract text.
- Integration validation: native/package/focused device/conformance evidence, independent review,
  combined repository and mixed-owner checkpoint.
- Shared-document integration owner: coordinator; worker owns only exact Metal version fact
  in owning contract and affected native/backend explanation/evidence.

Launch verified 2026-10-13: current root/scoped contracts explicitly require all four behaviors;
source confirms mismatches, predecessors Complete, no Metal writer. User authorizes bounded
repairs. Use clean context and isolated worktree, not new numerical architecture.
