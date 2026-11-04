# Documentation style profiles

## Purpose

These profiles make documentation expectations explicit without forcing every document to use the same shape. All documentation uses the common rules in [General style](general-style.md), then adds one profile based on its primary purpose.

The profiles explain how to present existing contracts and behavior. [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) remains the authoritative architecture contract, and the [documentation workflow](../documentation-rules.md) remains mandatory.

## How to use the profiles

1. Decide what the reader is trying to accomplish.
2. Read [General style](general-style.md).
3. Select the primary type from the table below.
4. Use [Example format](example-format.md) for examples.
5. Apply the clean-context review and validation workflow in the [documentation rules](../documentation-rules.md).

| Document purpose | Primary profile | Typical location |
|---|---|---|
| Teach a contributor how the project or an internal mechanism works | [Developer guide](developer-guide-style.md) | `docs/developer-guide/` |
| Teach a connected learning journey or help a user complete a task | [User guide](user-guide-style.md) and [example format](example-format.md) | `docs/tutorials/`, `docs/how-to/`, existing `docs/user-guide/`, `docs/examples/` |
| Explain architectural principles, boundaries, and rationale | [Architecture](architecture-style.md) | `docs/explanation/`, focused `docs/architecture/` explanations |
| Define or look up a Java/public API contract | [API and Javadoc](api-and-javadoc-style.md) | `docs/reference/`, Java Javadoc; old `docs/api/` paths retain incoming navigation |
| Teach backend integration or implementation | [Backend guide](backend-guide-style.md) | `docs/backend-guide/` |
| Coordinate executable implementation work | [Planning](planning-style.md) | `docs/planning/` |
| Record a significant decision and its consequences | [Decision record](decision-record-style.md) | `docs/design/decisions/` |

## Mixed documents

Choose the profile that matches the document's main reader outcome. For example, an architecture explanation may include a small Java snippet, but it remains an architecture document; use the architecture profile and only the example-contract rules needed from the API profile.

Do not use a mixed purpose to omit required content. If one document becomes both a tutorial and an architecture specification, split it into focused documents and cross-link them.

## Profile map

- [General style](general-style.md) — language, terminology, glossary, links, and illustrations shared by every document.
- [Developer guide](developer-guide-style.md) — concept teaching for contributors.
- [User guide](user-guide-style.md) — task completion for users.
- [Architecture](architecture-style.md) — principles, boundaries, rationale, scenarios, and diagrams.
- [API and Javadoc](api-and-javadoc-style.md) — precise callable contracts.
- [Backend guide](backend-guide-style.md) — backend lifecycle and integration.
- [Planning](planning-style.md) — executable plans and task specifications.
- [Decision record](decision-record-style.md) — context, options, decision, and consequences.
- [Example format](example-format.md) — reusable structure for complete, interpretable examples.
