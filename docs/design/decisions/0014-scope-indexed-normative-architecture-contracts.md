# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0014: [Scope](../../glossary.md#term-scope)-indexed [normative](../../glossary.md#term-normative) architecture contracts


## Editorial reading context

A contributor follows the root authority index to the contract owning the affected boundary. “Normative” means binding within that scope; an ADR explains a choice but does not independently change permission. Editorial responsibility for updating a contract is not ownership of an AutoCloseable resource. Preserve the recorded migration rather than using this ADR as a second rule source.

Use the [authority-aware review workflow](../../developer-guide/documentation-rules.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-09-22

## Context

The authoritative architecture had grown into one 62 KB root containing global invariants,
[dependency](../../glossary.md#term-dependency) rules, detailed [module](../../glossary.md#term-module) [ownership](../../glossary.md#term-ownership-meanings), specialized recurrent semantics, [backend](../../glossary.md#backend) execution,
and extension lifecycles. That monolith kept authority unambiguous, but it forced focused work to
load unrelated contracts and made exact task routing difficult.

The repository needed smaller [scope](../../glossary.md#term-scope)-indexed contracts without weakening authority or silently
changing a technical rule. Ordinary architecture explanations, plans, task briefs, source
comments, and tests could not become competing [normative](../../glossary.md#term-normative) sources.

## Decision drivers

- preserve every existing architecture rule, [ownership](../../glossary.md#term-ownership-meanings) boundary, [lifecycle](../../glossary.md#lifecycle), [dependency](../../glossary.md#term-dependency), and gate;
- keep one unambiguous authority index and global precedence rule;
- let contributors read the exact contract for an affected [module](../../glossary.md#term-module) or boundary;
- prevent overlapping scoped authority and duplicated detailed rules; and
- make missing or ambiguous [scope](../../glossary.md#term-scope) [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation).

## Options considered

### Keep the monolithic root

This preserves one physical file, but unrelated detail remains coupled and focused task packets
must continue routing readers through a large contract.

### Move details into non-[normative](../../glossary.md#term-normative) explanations

This shortens the root, but explanatory pages could not enforce detailed constraints. Treating
them as de facto authority would make precedence implicit and inconsistent.

### Duplicate important rules in the root and scoped contracts

This makes either location locally convenient, but two [normative](../../glossary.md#term-normative) copies can drift and leave a
reviewer unsure which text owns a later update.

### Incorporate six non-overlapping scoped contracts

Keep a compact authoritative root as the sole authority index. Retain global invariants and
[dependency](../../glossary.md#term-dependency) rules there, and explicitly incorporate exactly six files for detailed, disjoint
scopes.

## Decision

Synaptik adopts the six-way [scope](../../glossary.md#term-scope)-indexed structure.

[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is the authoritative root and sole authority index.
It explicitly incorporates these [normative](../../glossary.md#term-normative) files:

1. [Foundational modules](../../architecture/contracts/foundational-modules.md)
2. [Fixed recurrent scan](../../architecture/contracts/recurrent-scan.md)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#term-cumulative-scan))</span>
3. [Compiler and automatic differentiation](../../architecture/contracts/compiler-autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [automatic differentiation glossary](../../glossary.md#autograd))</span>
4. [Runtime, Prepare, and Engine](../../architecture/contracts/runtime-prepare-engine.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Engine glossary](../../glossary.md#term-engine))</span>
5. [Backend execution](../../architecture/contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend))</span>
6. [Extensions and training](../../architecture/contracts/extensions-training.md)

Each file is normative only for its stated non-overlapping scope, points back to the root, and
claims no independent authority. Root global invariants and [dependency](../../glossary.md#term-dependency) rules apply everywhere and
take precedence. Any overlap, contradiction, missing applicable scope, or ambiguity requires an
explicit architecture update; no contributor or agent chooses silently.

All other documentation, including this [architecture decision record](../../glossary.md#term-architecture-decision-record-adr) ([ADR](../../glossary.md#adr-ci-and-qualification)), remains explanatory
or coordinative and cannot override the contract.

The migration moves detailed text primarily verbatim, with mechanical heading, link, and context
edits. It changes no technical architecture rule, [module](../../glossary.md#term-module) edge, [ownership](../../glossary.md#term-ownership-meanings), [lifecycle](../../glossary.md#lifecycle), capability,
behavior, implementation status, or plan.

## Rationale

The selected structure retains a single authority [graph](../../glossary.md#graph) while reducing the amount of unrelated
material a focused task must read. Six cohesive domains are broad enough to avoid fragmented
micro-contracts and narrow enough to give every moved rule one [owner](../../glossary.md#term-owner). Keeping cross-cutting
invariants and [dependency direction](../../glossary.md#java-platform-and-build-tools) in the root makes precedence visible before a reader follows
a scoped link.

## Consequences

### Positive

- Focused task briefs can cite one root section and one exact scoped heading.
- Detailed rules keep [normative](../../glossary.md#term-normative) force without remaining in a monolith.
- The root provides a complete routing and former-heading destination table.
- Conflicts and missing [scope](../../glossary.md#term-scope) [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) instead of being resolved by document order or guesswork.

### Negative and risks

- Links to former root detail headings must migrate as affected documents are maintained.
- Contract changes now require checking both the root index/global rules and exactly one scoped
  owner.<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>
- Careless duplication can recreate competing [normative](../../glossary.md#term-normative) text, so [validation](../../glossary.md#term-validation) must detect it.

### Migration, [validation](../../glossary.md#term-validation), and follow-up

The migration validates every former top-level heading against its root or scoped destination,
compares key [invariant](../../glossary.md#term-invariant) language before and after the split, checks local links and anchors,
requires unique headings and balanced fences, and confirms exactly six scoped files. Navigation
and process documents are updated in the same change.

No [dependency](../../glossary.md#term-dependency) or build rule changes, so architecture-test source does not change and the suite is
not rerun solely for this authority-[distribution](../../glossary.md#term-distribution) refactor. No Java behavior or [API](../../glossary.md#term-application-programming-interface-api) changes, so no
Java tests or [Javadoc](../../glossary.md#term-javadoc) generation are required.

## Related documentation

- [Architecture root and scope index](../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([scope glossary](../../glossary.md#term-scope))</span>
- [Current architecture documentation](../../architecture/current-architecture-plan.md)
- [Documentation rules](../../developer-guide/documentation-rules.md)
- [Planning guide](../../planning/planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning))</span>
