# Architecture documentation style

## Purpose

Architecture documentation explains the principles, boundaries, and rationale of the architecture contract. It helps readers predict where responsibilities belong and understand why prohibited dependencies or lifecycle shortcuts are harmful.

Architecture explanations remain subordinate to [`ARCHITECTURE.md`](../../../ARCHITECTURE.md). Apply [General style](general-style.md) with this profile.

## Required content

- Open with an authority statement linking to the architecture contract.
- State the principle or boundary being explained.
- Identify the owning modules, data, decisions, and lifecycle stages.
- Show allowed and forbidden responsibilities or dependency directions.
- Explain the rationale and the failure mode prevented by the boundary.
- Include at least one concrete scenario that follows the boundary and, when useful, one counter-scenario that violates it.
- Use a small diagram for flows, dependency direction, ownership, or state transitions when prose alone is harder to follow.
- Describe consequences for implementation, testing, and future changes.
- Link to related ADRs, API documentation, and focused architecture pages.

A diagram must have a clear reading direction and nearby prose explaining the relationship. Scenarios should use real architecture vocabulary and must not invent unapproved modules or APIs.

## Avoid

- introducing a new architecture rule only in explanatory documentation;
- copying large sections of the contract without added explanation;
- class-level implementation tours that obscure the boundary;
- unlabeled aspirational designs presented as current architecture;
- diagrams with ambiguous arrows or mixed compile-time and runtime state; and
- rationale that says only “for cleanliness” without naming the concrete risk.

## Validation

- Compare every normative-sounding statement with `ARCHITECTURE.md`.
- Verify module ownership and dependency arrows against architecture tests where applicable.
- Walk each scenario through the documented lifecycle.
- Confirm diagrams distinguish compile, prepare, run, and per-run state correctly.
- Check links to the contract, ADRs, and adjacent explanations.

## Template

~~~markdown
# <Architectural concept>

> `ARCHITECTURE.md` is authoritative; this document explains its rule.

## Purpose and principle

## Mental model

```text
<small ownership, dependency, or lifecycle diagram>
```

## Responsibilities and boundaries

### Owned here

### Deliberately owned elsewhere

## Rationale

## Scenario that follows the boundary

## Counter-scenario and failure mode

## Implementation and testing consequences

## Related decisions and documentation
~~~
