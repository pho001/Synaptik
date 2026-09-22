# ADR 0014: Scope-indexed normative architecture contracts

## Status

Accepted — 2026-09-22

## Context

The authoritative architecture had grown into one 62 KB root containing global invariants,
dependency rules, detailed module ownership, specialized recurrent semantics, backend execution,
and extension lifecycles. That monolith kept authority unambiguous, but it forced focused work to
load unrelated contracts and made exact task routing difficult.

The repository needed smaller scope-indexed contracts without weakening authority or silently
changing a technical rule. Ordinary architecture explanations, plans, task briefs, source
comments, and tests could not become competing normative sources.

## Decision drivers

- preserve every existing architecture rule, ownership boundary, lifecycle, dependency, and gate;
- keep one unambiguous authority index and global precedence rule;
- let contributors read the exact contract for an affected module or boundary;
- prevent overlapping scoped authority and duplicated detailed rules; and
- make missing or ambiguous scope fail closed.

## Options considered

### Keep the monolithic root

This preserves one physical file, but unrelated detail remains coupled and focused task packets
must continue routing readers through a large contract.

### Move details into non-normative explanations

This shortens the root, but explanatory pages could not enforce detailed constraints. Treating
them as de facto authority would make precedence implicit and inconsistent.

### Duplicate important rules in the root and scoped contracts

This makes either location locally convenient, but two normative copies can drift and leave a
reviewer unsure which text owns a later update.

### Incorporate six non-overlapping scoped contracts

Keep a compact authoritative root as the sole authority index. Retain global invariants and
dependency rules there, and explicitly incorporate exactly six files for detailed, disjoint
scopes.

## Decision

Synaptik adopts the six-way scope-indexed structure.

[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is the authoritative root and sole authority index.
It explicitly incorporates these normative files:

1. [Foundational modules](../../architecture/contracts/foundational-modules.md)
2. [Fixed recurrent scan](../../architecture/contracts/recurrent-scan.md)
3. [Compiler and automatic differentiation](../../architecture/contracts/compiler-autograd.md)
4. [Runtime, Prepare, and Engine](../../architecture/contracts/runtime-prepare-engine.md)
5. [Backend execution](../../architecture/contracts/backend-execution.md)
6. [Extensions and training](../../architecture/contracts/extensions-training.md)

Each file is normative only for its stated non-overlapping scope, points back to the root, and
claims no independent authority. Root global invariants and dependency rules apply everywhere and
take precedence. Any overlap, contradiction, missing applicable scope, or ambiguity requires an
explicit architecture update; no contributor or agent chooses silently.

All other documentation, including this architecture decision record (ADR), remains explanatory
or coordinative and cannot override the contract.

The migration moves detailed text primarily verbatim, with mechanical heading, link, and context
edits. It changes no technical architecture rule, module edge, ownership, lifecycle, capability,
behavior, implementation status, or plan.

## Rationale

The selected structure retains a single authority graph while reducing the amount of unrelated
material a focused task must read. Six cohesive domains are broad enough to avoid fragmented
micro-contracts and narrow enough to give every moved rule one owner. Keeping cross-cutting
invariants and dependency direction in the root makes precedence visible before a reader follows
a scoped link.

## Consequences

### Positive

- Focused task briefs can cite one root section and one exact scoped heading.
- Detailed rules keep normative force without remaining in a monolith.
- The root provides a complete routing and former-heading destination table.
- Conflicts and missing scope fail closed instead of being resolved by document order or guesswork.

### Negative and risks

- Links to former root detail headings must migrate as affected documents are maintained.
- Contract changes now require checking both the root index/global rules and exactly one scoped
  owner.
- Careless duplication can recreate competing normative text, so validation must detect it.

### Migration, validation, and follow-up

The migration validates every former top-level heading against its root or scoped destination,
compares key invariant language before and after the split, checks local links and anchors,
requires unique headings and balanced fences, and confirms exactly six scoped files. Navigation
and process documents are updated in the same change.

No dependency or build rule changes, so architecture-test source does not change and the suite is
not rerun solely for this authority-distribution refactor. No Java behavior or API changes, so no
Java tests or Javadoc generation are required.

## Related documentation

- [Architecture root and scope index](../../../ARCHITECTURE.md)
- [Current architecture documentation](../../architecture/current-architecture-plan.md)
- [Documentation rules](../../developer-guide/documentation-rules.md)
- [Planning guide](../../planning/planning-guide.md)
