# Interactive TUI over shared CLI workflows

Status: Ready — user selected interactive TUI with retained CLI, and requested a shared ASCII/ANSI logo.

## Authorization and architecture boundary

The user expanded [0001](0001-live-model-training-and-terminal.md) from a training dashboard to
an interactive terminal application: menu, data/model/backend/training forms, live state,
events and safe stop. Keep `train`, `predict` and `inspect` usable for scripts.

Read [AGENTS.md](../../../../../AGENTS.md), the root architecture, [other tool scopes](../../../../architecture/contracts/backend-execution.md#other-tool-scopes)
and the public [Model/Training ownership contract](../../../../architecture/contracts/extensions-training.md#extensionsnn).
This is an outer presentation/controller change. It must not change Model arithmetic, native ABI,
Runtime execution, numerical ownership or inward module edges. No subprocess CLI invocation,
output parsing, private model state, second executor or hard interruption of an admitted call.

- **Depends on:** integrated real CLI workflows and repaired 58-test source boundary; current public owners.
- **Conflicts with:** other writers of CLI progress, controller, terminal or command parsing.
- **Parallel group:** `interactive-tui`; terminal/form/view/brand and controller/shared-workflow scopes.
- **Common base revision:** recorded in the clean-worktree receipt before writers start.
- **Integration order:** terminal/forms/view/brand, then shared workflow/controller.
- **Integration owner:** coordinator; shared documentation, notices and source selection remain coordinator-owned.
- **Validation:** parent-only actual keyboard/PTY/forms/training/prediction/inspection/error/resize/interrupt proof and focused behavior tests; writers run no checks.

## Two presentation owners, one computation path

`CsvRegressionWorkflow` remains the common application workflow. CLI obtains its existing
progress owner; TUI injects a detached monitor and owns one terminal/event loop. Training runs
on one worker which creates and closes Model, Engine and batch storage normally. UI refresh
reads detached scalar snapshots, never model/native state. An INT/stop request sets a flag;
no worker interruption or `Future.cancel(true)` is permitted.

The menu exposes training, prediction, inspection and quit. Forms build the same command options
and reuse the same validation. Loaded models visibly disable new-model seed/geometry controls;
no displayed control may be silently ignored. Prediction in TUI uses an explicit output file so
CSV does not corrupt the screen. Inspection presents public detached source metadata.

Training supports safe stop between calls. A currently admitted save can complete; late
cancellation reports the already-saved checkpoint honestly. Prediction/inspection may finish
their already admitted operation before returning/exit; do not invent partial-operation rollback.

## Terminal and presentation

Use the pinned JLine FFM provider and matching qualified POSIX stdin/stderr TTY identity. A full
TUI additionally requires alternate-screen/cursor capabilities; unsupported or redirected use
reports the prerequisite and leaves CLI/plain commands available. Own raw input, alternate screen,
signals and cursor in one outer terminal resource and restore all of them on every reachable exit.

Menu/form keys must be discoverable. Include field editing/navigation, start/back, safe training
stop and quit; the UI stays responsive while the worker runs. Requery size and fit rows/columns.
Keep the 500 ms default refresh, real epoch/time/step/payload rates, bounded actual loss history
and a bounded event panel. No fake curves or nonfinite-value repair.

Use the approved compact connection-node SYNAPTIK logo in both frontends. ASCII is always valid;
ANSI color uses cyan nodes, dim links and a bold default-color name without a painted background.
Respect NO_COLOR/terminal capabilities and use the name alone when space is limited. Logo
snapshots/source copies are not permanent tests.

## Acceptance

- Actual TUI menu and form navigation start real training; saved files predict real values.
- Existing CLI behavior remains exercised, including plain/noTTY data separation.
- PTY proof covers keyboard, Unicode/path editing where supported, forms/errors, default refresh,
  status/loss/events, resize, completion, stop/INT and terminal restoration.
- A terminal/presentation failure requests stop and waits for the admitted computation to close;
  it never closes Model/Engine from the UI thread or interrupts the worker.
- Meaningful deterministic tests cover editing/state/geometry/lifecycle boundaries and existing
  numerical/data invariants; no wording/default/ASCII/wiring/mock-echo tests.
- Independent source review, current guide/README/backlinks/source inventory and exact exercised
  evidence are updated before closure. Performance comparison follows the completed interface.
