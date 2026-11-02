#ifndef SYNAPTIK_GENERAL_SCAN_BINDING_KERNELS_H
#define SYNAPTIK_GENERAL_SCAN_BINDING_KERNELS_H

#include "synaptik_general_scan.h"

typedef struct {
    uint64_t elementCount, sourceBase, destinationBase;
    uint32_t width, mode, gridWidth, gridHeight;
} SynaptikMetalScanBindMeta;
_Static_assert(sizeof(SynaptikMetalScanBindMeta) == 40, "Scan binding metadata size");

/* Raw movement only. Depends on existing exact load_word/store_word and common Scan controls.
 * mode0 publishes present values, mode1 replaces carried roots (absence writes raw +0), mode2
 * updates accumulators only for present contributions. Inactive carry cells remain raw identity.
 * Buffers source0/destination1/meta2/oldCarry3/control28/step29; ordinary custom_grid geometry.
 */
static NSString *const SynaptikGeneralScanBindingKernelSource =
@"\nstruct ScanBindMeta { ulong elementCount; ulong sourceBase; ulong destinationBase; uint width; uint mode; uint gridWidth; uint gridHeight; };\n"
"kernel void scan_bind_raw(device const uchar *source [[buffer(0)]], device uchar *destination [[buffer(1)]], constant ScanBindMeta &m [[buffer(2)]], device const uchar *oldCarry [[buffer(3)]], device const ulong *control [[buffer(28)]], constant ScanStepControl &step [[buffer(29)]], uint3 gid [[thread_position_in_grid]]) {\n"
" ulong linear=ulong(gid.x)+ulong(m.gridWidth)*(ulong(gid.y)+ulong(m.gridHeight)*ulong(gid.z)); if(linear>=m.elementCount)return;\n"
" bool present=scan_control_output(control,step,linear);\n"
" if(m.mode!=0u){ bool eligible; if(step.rowAxis<0) eligible=scan_control_any(control,0xffffffffu); else eligible=control[1]!=0ul&&step.rowInner!=0ul&&scan_control_row(control,0xffffffffu,(linear/step.rowInner)%control[1]); if(!eligible){store_word(destination,m.destinationBase+linear,m.width,load_word(oldCarry,linear,m.width));return;} }\n"
" if(!present){ if(m.mode==1u)store_word(destination,m.destinationBase+linear,m.width,0ul); else if(m.mode==2u)store_word(destination,m.destinationBase+linear,m.width,load_word(oldCarry,linear,m.width)); return; }\n"
" ulong word=load_word(source,m.sourceBase+linear,m.width); store_word(destination,m.destinationBase+linear,m.width,word);\n"
"}\n";

#endif
