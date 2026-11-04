# Documentation style profiles

## Purpose

These profiles make documentation expectations explicit without forcing every document to use the same shape. All documentation uses the common rules in [General style](general-style.md), then adds one profile based on its primary purpose.

The profiles explain how to present existing contracts and behavior. [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) remains the authoritative [architecture contract](../../glossary.md#architecture-contract), and the [documentation workflow](../documentation-rules.md) remains mandatory.

A documentation **profile** is a writing checklist selected by the reader's goal, not a [runtime](../../glossary.md#term-runtime)
configuration or a numerical mode. A **contract** states behavior or responsibilities that must
be preserved; a profile tells you how to explain that contract. Start with the table's purpose
column, not with the name of the directory where a file happens to live.

## How to use the profiles

1. Decide what the reader is trying to accomplish.
2. Read [General style](general-style.md).
3. Select the primary type from the table below.
4. Use [Example format](example-format.md) for examples.
5. Apply the clean-context review and [validation](../../glossary.md#term-validation) workflow in the [documentation rules](../documentation-rules.md).

| Document purpose | Primary profile | Typical location |
|---|---|---|
| Teach a contributor how the project or an internal mechanism works | [Developer guide](developer-guide-style.md) | `docs/developer-guide/` |
| Teach a connected learning journey or help a user complete a task | [User guide](user-guide-style.md) and [example format](example-format.md) | `docs/tutorials/`, `docs/how-to/`, existing `docs/user-guide/`, `docs/examples/` |
| Explain architectural principles, boundaries, and rationale | [Architecture](architecture-style.md) | `docs/explanation/`, focused `docs/architecture/` explanations |
| Define or look up a Java/public [API](../../glossary.md#term-application-programming-interface-api) contract | [API and Javadoc](api-and-javadoc-style.md) | `docs/reference/`, Java [Javadoc](../../glossary.md#term-javadoc); old `docs/api/` paths retain incoming navigation |
| Teach [backend](../../glossary.md#backend) integration or implementation | [Backend guide](backend-guide-style.md) | `docs/backend-guide/` |
| [Coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) executable implementation work | [Planning](planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning))</span> | `docs/planning/` |
| Record a significant decision and its consequences | [Decision record](decision-record-style.md) | `docs/design/decisions/` |

## Mixed documents

Choose the profile that matches the document's main reader outcome. For example, an architecture explanation may include a small Java snippet, but it remains an architecture document; use the architecture profile and only the example-contract rules needed from the [API](../../glossary.md#term-application-programming-interface-api) profile.

Do not use a mixed purpose to omit required content. If one document becomes both a tutorial and an architecture specification, split it into focused documents and cross-link them.

## Profile map

- [General style](general-style.md) — language, terminology, glossary, links, and illustrations shared by every document.
- [Developer guide](developer-guide-style.md) — concept teaching for contributors.
- [User guide](user-guide-style.md) — task completion for users.
- [Architecture](architecture-style.md) — principles, boundaries, rationale, scenarios, and diagrams.
- [API and Javadoc](api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../glossary.md#term-javadoc))</span> — precise callable contracts.
- [Backend guide](backend-guide-style.md) — [backend](../../glossary.md#backend) [lifecycle](../../glossary.md#lifecycle) and integration.
- [Planning](planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning))</span> — executable plans and task specifications.
- [Decision record](decision-record-style.md) — context, options, decision, and consequences.
- [Example format](example-format.md) — reusable structure for complete, interpretable examples.
