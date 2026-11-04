# Engine: compile, prepare and run

Use ordinary Engine for reusable execution, or explicit composition when you must select opened CPU/Metal owners. Compilation retains logical recipes; prepared handles and result leases carry separate lifetimes.

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)

Compile describes the computation, prepare constructs reusable execution recipes, and run
connects one set of stored inputs to those recipes. A result lease keeps that invocation's
completed resources alive; materializing a result makes an independent host copy. Reuse the
preparation for repeated runs, but close each result when its selected outputs are no longer needed.

## Choose a topic

- [CPU integration and native ownership](cpu-integration.md)
- [Ordinary Engine compile, prepare and run lifecycle](lifecycle.md)
- [Advanced Engine lifecycle](advanced.md)
- [Explicit CPU and Metal composition](composition.md)
- [CPU limitations and development compatibility](limitations.md)
