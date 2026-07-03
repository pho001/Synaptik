# Decision record style

## Purpose

An architecture decision record (ADR) preserves why a significant decision was needed, which credible options were considered, what was chosen, and what consequences follow. It allows future contributors to evaluate the decision without reconstructing old discussions.

An ADR records a decision; it does not override [`ARCHITECTURE.md`](../../../ARCHITECTURE.md). When a decision changes architecture, update the contract and other required artifacts in the same overall change.

## Required content

- **Status and date:** proposed, accepted, superseded, or another defined state, with timing.
- **Context:** the concrete problem, forces, constraints, and evidence that require a decision.
- **Decision drivers:** the criteria used to compare options.
- **Options considered:** credible alternatives, including the status quo when relevant.
- **Decision:** the selected option stated unambiguously.
- **Rationale:** why it best satisfies the drivers and architecture constraints.
- **Consequences:** positive, negative, operational, compatibility, testing, migration, and follow-up effects.
- **Links:** affected contract sections, focused architecture docs, tasks, tests, and superseded ADRs.

Use scenarios or a compact comparison table when tradeoffs are otherwise hard to see.

## Avoid

- recording a decision with no alternatives or context;
- presenting preference as evidence;
- rewriting history after implementation without identifying retrospective evidence;
- hiding costs, migration, or rejected options;
- placing implementation task checklists in the ADR; and
- treating the ADR as authority when the architecture contract says otherwise.

## Validation

- Confirm the decision and alternatives address the same stated problem.
- Check rationale against explicit decision drivers.
- Verify consequences include disadvantages and follow-up obligations.
- Ensure architecture, focused docs, tests, and plans are updated together when required.
- Validate links, status, date, and supersession relationships.

## Template

```markdown
# ADR <ID>: <Decision title>

## Status

<Proposed | Accepted | Superseded> — <date>

## Context

## Decision drivers

## Options considered

### Option 1: ...

### Option 2: ...

## Decision

## Rationale

## Consequences

### Positive

### Negative and risks

### Migration, testing, and follow-up

## Related documentation
```
