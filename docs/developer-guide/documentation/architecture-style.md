# Architecture documentation style

## Purpose

Architecture documentation explains the principles, boundaries, and rationale of the [architecture contract](../../glossary.md#architecture-contract). It helps readers predict where responsibilities belong and understand why prohibited dependencies or [lifecycle](../../glossary.md#lifecycle) shortcuts are harmful.

Architecture explanations remain subordinate to [`ARCHITECTURE.md`](../../../ARCHITECTURE.md). Apply [General style](general-style.md) with this profile.

A **boundary** separates responsibilities: who may decide something, depend on something, or own
a resource. [Ownership](../../glossary.md#term-ownership-meanings) can mean authority over a decision,
private mutable state, or responsibility for a resource's release. State which one applies;
assigning a graph region to CPU is not the same action as transferring a native handle.
A **lifecycle** is the sequence from creation through use to [release](../../glossary.md#term-release). Explain those relationships
before listing class names. An **[architecture decision record](../../glossary.md#term-architecture-decision-record-adr) ([ADR](../../glossary.md#adr-ci-and-qualification))** preserves the reasons for a
choice; it is not another source of architecture authority.

## Required content

- Open with an authority statement linking to the [architecture contract](../../glossary.md#architecture-contract).
- State the principle or boundary being explained.
- Identify the owning [modules](../../glossary.md#term-module), data, decisions, and [lifecycle](../../glossary.md#lifecycle) stages.
- Show allowed and forbidden responsibilities or [dependency](../../glossary.md#term-dependency) directions.
- Explain the rationale and the failure mode prevented by the boundary.
- Include at least one concrete scenario that follows the boundary and, when useful, one counter-scenario that violates it.
- Use a small diagram for flows, [dependency direction](../../glossary.md#term-dependency), [ownership](../../glossary.md#term-ownership-meanings), or state transitions when prose alone is harder to follow.
- Describe consequences for implementation, testing, and future changes.
- Link to related ADRs, [API](../../glossary.md#term-application-programming-interface-api) documentation, and focused architecture pages.

A diagram must have a clear reading direction and nearby prose explaining the relationship. Scenarios should use real architecture vocabulary and must not invent unapproved modules or APIs.

## Teach one collaboration, not an inventory

For staged preparation, name shared Prepare as the caller of backend analysis.
Explain the input partition, the returned requirements and the later slot-assignment
response before drawing arrows. Label an arrow as a call, dependency or transfer;
those relationships are not interchangeable. A dependency arrow does not imply
that the target closes the source's resources.

Use [staged preparation](../../backend-guide/partition-preparer.md) for the concrete
scenario and [architecture tests](../architecture-tests.md) for the narrower question
of which forbidden relationships are enforced. Mermaid diagrams need `accTitle`
and `accDescr` so their relationship is available without relying only on the picture.

## Avoid

- introducing a new architecture rule only in explanatory documentation;
- copying large sections of the contract without added explanation;
- class-level implementation tours that obscure the boundary;
- unlabeled aspirational designs presented as current architecture;
- diagrams with ambiguous arrows or mixed [compile](../../glossary.md#compile)-time and [runtime](../../glossary.md#term-runtime) state; and
- rationale that says only “for cleanliness” without naming the concrete risk.

## [Validation](../../glossary.md#term-validation)

- Compare every [normative](../../glossary.md#term-normative)-sounding statement with `ARCHITECTURE.md`.
- Verify [module](../../glossary.md#term-module) [ownership](../../glossary.md#term-ownership-meanings) and [dependency](../../glossary.md#term-dependency) arrows against architecture tests where applicable.
- Walk each scenario through the documented [lifecycle](../../glossary.md#lifecycle).
- Confirm diagrams distinguish [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), [run](../../glossary.md#term-run), and per-[run state](../../glossary.md#run-state--runstate) correctly.
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
