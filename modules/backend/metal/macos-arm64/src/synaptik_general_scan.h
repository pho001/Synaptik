#ifndef SYNAPTIK_GENERAL_SCAN_H
#define SYNAPTIK_GENERAL_SCAN_H

#include <stdint.h>
#include <stddef.h>

/* Common authenticated control ABI. Ordinary entry points are unchanged. Controlled entry
 * points append scanControl at buffer(28), scanStep at buffer(29), after all ordinary bindings.
 * No control pointer is optional on a controlled entry. Host independently authenticates every
 * scope/row mapping and proves buffer bounds before encoding. All offsets below are ulong WORD
 * offsets, not bytes. The first ten control words are: original time, batch, root count, scope
 * count, words per root set, membership offset, presence offset, lengths offset, length-aware,
 * presence row count (1 for all-valid; B for length-aware). Memberships are scope-major bitsets;
 * presence is original-row-major bitsets. Lengths are prevalidated signed INT64 words.
 * All mutable storage is invocation-owned. No numeric tensor value is structural presence.
 */
/* First-failure diagnostics occupy fixed words 10..14; membership starts at 15.
 * Only the synchronous host validator writes them. No payload or device address escapes. */
static void synaptik_scan_failure(const uint64_t *control, uint64_t category,
        uint64_t operand, uint64_t ordinal) {
    if (control == NULL || control[10] != 0U) return;
    uint64_t *record = (uint64_t *)control;
    record[10] = category; record[11] = control[0];
    record[13] = operand; record[14] = ordinal;
}

typedef struct {
    uint64_t rowInner;
    uint32_t scope;
    uint32_t leftScope;
    uint32_t rightScope;
    int32_t rowAxis;
    uint32_t reserved[2];
    int32_t inputRowAxes[4];
    uint64_t inputRowInner[4];
    int32_t outputRowAxes[5];
    uint32_t outputScopes[5];
    uint64_t outputRowInner[5];
    uint64_t outputElements[5];
    uint32_t outputCount;
    uint32_t outputReserved;
} SynaptikMetalScanStepControl;
_Static_assert(sizeof(SynaptikMetalScanStepControl) == 208, "ScanStepControl size");
_Static_assert(_Alignof(SynaptikMetalScanStepControl) == 8, "ScanStepControl alignment");
_Static_assert(offsetof(SynaptikMetalScanStepControl, inputRowAxes) == 32, "Scan input axes offset");
_Static_assert(offsetof(SynaptikMetalScanStepControl, inputRowInner) == 48, "Scan input inner offset");
_Static_assert(offsetof(SynaptikMetalScanStepControl, outputRowAxes) == 80, "Scan output axes offset");
_Static_assert(offsetof(SynaptikMetalScanStepControl, outputScopes) == 100, "Scan output scopes offset");
_Static_assert(offsetof(SynaptikMetalScanStepControl, outputRowInner) == 120, "Scan output inner offset");
_Static_assert(offsetof(SynaptikMetalScanStepControl, outputElements) == 160, "Scan output elements offset");
_Static_assert(offsetof(SynaptikMetalScanStepControl, outputCount) == 200, "Scan output count offset");

/* Host preflights use the same authenticated control words as shaders. Null controls are only
 * the existing ordinary non-Scan path; controlled callers always supply both pointers. */
static BOOL synaptik_scan_control_row(const uint64_t *control, uint32_t scope, uint64_t row) {
    if (control == NULL) return YES;
    if (row >= control[9]) return NO;
    if (control[8] != 0U && (row >= control[1] || control[0] >= control[control[7] + row])) return NO;
    if (scope == UINT32_MAX) return YES;
    if (scope >= control[3]) return NO;
    uint64_t members = control[5] + (uint64_t)scope * control[4];
    uint64_t presence = control[6] + row * control[4];
    for (uint64_t word = 0U; word < control[4]; word++)
        if ((control[members + word] & control[presence + word]) != 0U) return YES;
    return NO;
}
static BOOL synaptik_scan_control_any(const uint64_t *control, uint32_t scope) {
    if (control == NULL) return YES;
    for (uint64_t row = 0U; row < control[9]; row++)
        if (synaptik_scan_control_row(control, scope, row)) return YES;
    return NO;
}
static BOOL synaptik_scan_control_output_scope(const uint64_t *control,
        const SynaptikMetalScanStepControl *step, uint32_t scope, uint64_t linear) {
    if (control == NULL || step == NULL) return YES;
    if (step->rowAxis < 0) return synaptik_scan_control_any(control, scope);
    if (control[1] == 0U || step->rowInner == 0U) return NO;
    return synaptik_scan_control_row(control, scope, (linear / step->rowInner) % control[1]);
}
static BOOL synaptik_scan_control_output(const uint64_t *control,
        const SynaptikMetalScanStepControl *step, uint64_t linear) {
    return synaptik_scan_control_output_scope(control, step, step == NULL ? UINT32_MAX : step->scope, linear);
}
static BOOL synaptik_scan_control_operand(const uint64_t *control,
        const SynaptikMetalScanStepControl *step, uint32_t role, uint64_t linear) {
    if (control == NULL || step == NULL) return YES;
    if (role >= 4U) return NO;
    if (step->inputRowAxes[role] < 0) return synaptik_scan_control_any(control, step->scope);
    if (control[1] == 0U || step->inputRowInner[role] == 0U) return NO;
    return synaptik_scan_control_row(control, step->scope,
            (linear / step->inputRowInner[role]) % control[1]);
}

/* Independent of DataMeta and all floating helpers; assemble before controlled components.

 * scan_control_output uses a canonical logical output ordinal, never a physical address.
 * scan_control_operand uses an original logical operand ordinal and exact semantic ROW facts.
 * Per-row presence and lengths are tested before ANY guarded data/index load. A SHARED scope is
 * the disjunction over eligible original rows. UINT_MAX is unconditional structural presence;
 * an empty declared scope is absent. For all-valid bodies rowCount is one even for empty tensors.
 */
static NSString *const SynaptikGeneralScanKernelSource =
@"\nstruct ScanStepControl { ulong rowInner; uint scope; uint leftScope; uint rightScope; int rowAxis; uint reserved[2]; int inputRowAxes[4]; ulong inputRowInner[4]; int outputRowAxes[5]; uint outputScopes[5]; ulong outputRowInner[5]; ulong outputElements[5]; uint outputCount; uint outputReserved; };\n"
"inline bool scan_control_row(device const ulong *control, uint scope, ulong row) {\n"
"  if (row >= control[9]) return false;\n"
"  if (control[8] != 0ul && (row >= control[1] || control[0] >= control[control[7] + row])) return false;\n"
"  if (scope == 0xffffffffu) return true;\n"
"  if (ulong(scope) >= control[3]) return false;\n"
"  ulong members = control[5] + ulong(scope) * control[4];\n"
"  ulong presence = control[6] + row * control[4];\n"
"  for (ulong word = 0ul; word < control[4]; ++word) if ((control[members + word] & control[presence + word]) != 0ul) return true;\n"
"  return false;\n"
"}\n"
"inline bool scan_control_any(device const ulong *control, uint scope) {\n"
"  for (ulong row = 0ul; row < control[9]; ++row) if (scan_control_row(control, scope, row)) return true;\n"
"  return false;\n"
"}\n"
"inline bool scan_control_output_scope(device const ulong *control, constant ScanStepControl &step, uint scope, ulong linear) {\n"
"  if (step.rowAxis < 0) return scan_control_any(control, scope);\n"
"  if (control[1] == 0ul || step.rowInner == 0ul) return false;\n"
"  return scan_control_row(control, scope, (linear / step.rowInner) % control[1]);\n"
"}\n"
"inline bool scan_control_output_at(device const ulong *control, constant ScanStepControl &step, uint output, ulong linear) {\n"
" if(output>=step.outputCount||output>=5u||linear>=step.outputElements[output])return false;\n"
" if(step.outputRowAxes[output]<0)return scan_control_any(control,step.outputScopes[output]);\n"
" if(control[1]==0ul||step.outputRowInner[output]==0ul)return false;\n"
" return scan_control_row(control,step.outputScopes[output],(linear/step.outputRowInner[output])%control[1]);\n"
"}\n"
"inline bool scan_control_any_output(device const ulong *control, constant ScanStepControl &step) {\n"
" for(uint output=0u;output<step.outputCount&&output<5u;++output)if(step.outputElements[output]!=0ul&&scan_control_any(control,step.outputScopes[output]))return true;\n"
" return false;\n"
"}\n"
"inline bool scan_control_output(device const ulong *control, constant ScanStepControl &step, ulong linear) { return scan_control_output_scope(control, step, step.scope, linear); }\n"
"inline bool scan_control_contributor(device const ulong *control, constant ScanStepControl &step, ulong row) { return scan_control_row(control, step.scope, row); }\n"
"inline bool scan_control_operand(device const ulong *control, constant ScanStepControl &step, uint role, ulong linear) {\n"
"  if (role >= 4u) return false;\n"
"  if (step.inputRowAxes[role] < 0) return scan_control_any(control, step.scope);\n"
"  if (control[1] == 0ul || step.inputRowInner[role] == 0ul) return false;\n"
"  return scan_control_row(control, step.scope, (linear / step.inputRowInner[role]) % control[1]);\n"
"}\n"
"inline uint scan_control_merge(device const ulong *control, constant ScanStepControl &step, ulong linear) {\n"
"  return (scan_control_output_scope(control, step, step.leftScope, linear) ? 1u : 0u) | (scan_control_output_scope(control, step, step.rightScope, linear) ? 2u : 0u);\n"
"}\n";

#endif
