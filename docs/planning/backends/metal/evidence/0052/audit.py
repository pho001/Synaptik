#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import importlib.util
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
metal=(ROOT/"oracle.metal").read_text()
host=(ROOT/"oracle.mm").read_text()
runner=(ROOT/"run-once.py").read_text()
validator=(ROOT/"validate.py").read_text()
fail=[]

def need(condition,message):
    if not condition: fail.append(message)

def extract_vector(name):
    match=re.search(rf"const std::vector<uint32_t> {name} = \{{(.*?)\}};",host,re.S)
    need(match is not None,f"missing vector {name}")
    return [] if match is None else [int(x,16) for x in re.findall(r"0x([0-9a-fA-F]+)u",match.group(1))]

spec=importlib.util.spec_from_file_location("task0052_validator",ROOT/"validate.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
for host_name,validator_name in (("pairLeft","PAIR_LEFT"),("pairRight","PAIR_RIGHT"),
                                 ("broadcastLeft","BROADCAST_LEFT"),("broadcastRight","BROADCAST_RIGHT"),
                                 ("scalars","SCALARS"),("aggregateWords","AGGREGATE")):
    need(extract_vector(host_name)==getattr(module,validator_name),f"corpus mismatch {host_name}")

functions=["cmp_gt","cmp_ge","cmp_lt","cmp_le","cmp_eq","cmp_ne","tensor_min","tensor_max",
           "scalar_min","scalar_max","clamp_fused","reduction_min","reduction_max","scan_sum","scan_prod"]
for name in functions:
    need(re.search(rf"(?:COMPARISON_KERNEL|BINARY_EXTREME_KERNEL|SCALAR_EXTREME_KERNEL|REDUCTION_KERNEL|SCAN_KERNEL)\({name},|kernel void {name}\(",metal) is not None,
         f"missing Metal entry point {name}")
    need(f'@"{name}"' in host,f"pipeline list missing {name}")
need(host.count("MTLCreateSystemDefaultDevice()") == 1,"device creation is not singular")
need(host.count("[queue commandBuffer]") == 1,"command buffer creation is not singular")
need(host.count("[commandBuffer commit]") == 1,"command buffer commit is not singular")
need(host.count("[commandBuffer waitUntilCompleted]") == 1,"command wait is not singular")
need(host.count("newCommandQueue") == 1,"command queue creation is not singular")
need(host.index("newComputePipelineStateWithFunction") < host.index("[queue commandBuffer]"),"pipelines not compiled before command buffer")
need("options.mathMode = MTLMathModeSafe;" in host,"safe math mode absent")
need("fastMathEnabled = YES" not in host and "MTLMathModeFast" not in host and "MTLMathModeRelaxed" not in host,"unsafe math enabled")
need("linear_id(uint3 gid" in metal and "gridWidth" in metal and "gridHeight" in metal,"uint3 full-grid mapping absent")
need("checkedMultiply" in host and "metadata product overflow" in host,"checked geometry absent")
need("if (nan_bits(left)) return left;" in metal and "if (nan_bits(right)) return right;" in metal,"NaN propagation absent")
need("maximum ? ((left & right)" in metal and "((left | right)" in metal,"signed-zero extrema rules absent")
need("bool hasContributor = false" in metal and "accumulator = value" in metal,"exact singleton scan path absent")
need("for (ulong contributor = 1ul" in metal,"deterministic reduction traversal absent")
need("range(1,len(sequence))" in validator and "tree_results(sequence[:split],product)" in validator,
     "ordered parenthesization enumeration absent")
need("operand_alternatives" in validator and "result_alternatives" in validator,"DAZ/FTZ closure absent")
need("NAN = \"NAN\"" in validator and "is_nan(actual)" in validator,"NaN class validation absent")
need("0x00000000u,0x80000000u" in host and "0x80000000u,0x00000000u" in host,"zero-order witnesses absent")
for token in ("single_pos_zero","single_neg_zero","single_pos_sub","single_neg_sub","single_pos_qnan","single_pos_snan","single_neg_qnan","single_neg_snan"):
    need(token in host and token in validator,f"missing edge witness {token}")
for token in ("axis2_inc_fwd","axis2_exc_fwd","axis2_inc_rev","axis2_exc_rev","axis1_inc_fwd"):
    need(token in host and token in validator,f"missing scan form {token}")
need(host.count("clampBounds") >= 2 and len(module.CLAMPS)==8,"clamp corpus incomplete")
need(len(module.PAIR_LEFT)==24 and len(module.PAIR_RIGHT)==24,"pair corpus incomplete")
need(len(module.AGGREGATE)==48,"aggregate corpus incomplete")
need((ROOT/"oracle").exists(),"host executable absent")
need(runner.count("subprocess.run(")==1,"capture wrapper does not invoke oracle exactly once")
need("capture_output=True" in runner and "check=False" in runner,"capture wrapper controls absent")
for path in (ROOT/"oracle.metal",ROOT/"oracle.mm",ROOT/"validate.py",ROOT/"run-once.py",ROOT/"oracle"):
    print(f"SHA256 {path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}")
if fail:
    for item in fail: print("AUDIT_FAIL="+item)
    print(f"AUDIT=FAIL FAILURES={len(fail)}")
    raise SystemExit(1)
print("AUDIT=PASS FAILURES=0")
