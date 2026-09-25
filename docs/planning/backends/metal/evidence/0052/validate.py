#!/usr/bin/env python3
from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import struct
import sys
from pathlib import Path

NAN = "NAN"
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000

PAIR_LEFT = [
    0x00000000,0x80000000,0x00000001,0x00000000,0x80000001,0x80000000,
    0x007fffff,0x00800000,0x807fffff,0x80800000,0x3f800000,0x3f800000,
    0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0x7fc12345,0x3f800000,
    0x7fa12345,0xbf800000,0xffc54321,0x00000000,0x7fc12345,0x40000000]
PAIR_RIGHT = [
    0x80000000,0x00000000,0x00000000,0x00000001,0x80000000,0x80000001,
    0x00800000,0x007fffff,0x80800000,0x807fffff,0x3f800000,0x3f800001,
    0xbf7fffff,0x7f800000,0xff800000,0xff800000,0x3f800000,0x7fc12345,
    0xbf800000,0x7fa12345,0x00000000,0xffc54321,0xffc54321,0xc0000000]
BROADCAST_LEFT = [0x3f800000,0x40000000,0x40400000,0xbf800000,0xc0000000,0xc0400000]
BROADCAST_RIGHT = [
    0x00000000,0x3f800000,0x40800000,0x40000000,0x40000000,0x40000000,
    0xc0000000,0xbf800000,0x00000000,0x41200000,0xc1200000,0x3f000000]
SCALARS = [0x00000000,0x80000000,0x00000001,0x80000001,0x3f800000,0xbf800000,0x7f800000,0x7fc12345]
CLAMPS = [
    (0xbf800000,0x3f800000),(0x80000000,0x00000000),
    (0x00000000,0x80000000),(0x00000001,0x3f800000),
    (0xbf800000,0x80000001),(0x7fc12345,0x3f800000),
    (0xbf800000,0x7fc12345),(0x7f800000,0x7f800000)]
AGGREGATE = [
    0x00000000,0x80000000,0x00000001,0x80000001,0x00800000,0x80800000,0x3f800000,0xbf800000,
    0x80000000,0x00000000,0x007fffff,0x807fffff,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
    0x7fc12345,0x3f800000,0xbf800000,0x7fa12345,0x00000000,0x80000000,0x40000000,0xc0000000,
    0xbf800000,0x3f800000,0x80800000,0x00800000,0x80000001,0x00000001,0x80000000,0x00000000,
    0xff800000,0x7f800000,0xff7fffff,0x7f7fffff,0x807fffff,0x007fffff,0x00000000,0x80000000,
    0xc0000000,0x40000000,0x80000000,0x00000000,0xffc54321,0xbf800000,0x3f800000,0xffa54321]
REDUCTION_EDGES = {
    "zero_pos_neg":[0x00000000,0x80000000], "zero_neg_pos":[0x80000000,0x00000000],
    "single_pos_zero":[0x00000000], "single_neg_zero":[0x80000000],
    "single_pos_sub":[0x00000001], "single_neg_sub":[0x80000001],
    "single_pos_qnan":[0x7fc12345], "single_pos_snan":[0x7fa12345],
    "single_neg_qnan":[0xffc54321], "single_neg_snan":[0xffa54321]}

DIRECT_CANDIDATES = [
    "direct_cmp_gt","direct_cmp_ge","direct_cmp_lt","direct_cmp_le","direct_cmp_eq","direct_cmp_ne",
    "direct_tensor_min","direct_tensor_max","direct_scalar_min","direct_scalar_max","direct_clamp",
    "composed_mpsgraph_clamp","direct_reduction_min","direct_reduction_max","direct_scan_sum","direct_scan_prod"]
CUSTOM_CANDIDATES = [
    "custom_cmp_gt","custom_cmp_ge","custom_cmp_lt","custom_cmp_le","custom_cmp_eq","custom_cmp_ne",
    "custom_tensor_min","custom_tensor_max","custom_scalar_min","custom_scalar_max","custom_clamp_fused",
    "composed_custom_clamp","custom_reduction_min","custom_reduction_max","custom_scan_sum","custom_scan_prod"]

def is_nan(word: int) -> bool:
    return (word & 0x7f800000) == 0x7f800000 and (word & 0x007fffff) != 0

def is_inf(word: int) -> bool:
    return (word & 0x7fffffff) == 0x7f800000

def is_zero(word: int) -> bool:
    return (word & 0x7fffffff) == 0

def is_subnormal(word: int) -> bool:
    return (word & 0x7f800000) == 0 and (word & 0x007fffff) != 0

def ordered_key(word: int) -> int:
    return (~word) & 0xffffffff if word & 0x80000000 else word ^ 0x80000000

def eq_word(left: int, right: int) -> bool:
    if is_nan(left) or is_nan(right): return False
    return (is_zero(left) and is_zero(right)) or left == right

def less_word(left: int, right: int) -> bool:
    return not is_nan(left) and not is_nan(right) and not eq_word(left, right) and ordered_key(left) < ordered_key(right)

def compare_word(left: int, right: int, op: str) -> int:
    values = {
        "gt": less_word(right,left), "ge": less_word(right,left) or eq_word(left,right),
        "lt": less_word(left,right), "le": less_word(left,right) or eq_word(left,right),
        "eq": eq_word(left,right), "ne": not eq_word(left,right)}
    return int(values[op])

def extreme_word(left: int, right: int, maximum: bool):
    if is_nan(left): return NAN
    if is_nan(right): return NAN
    if is_zero(left) and is_zero(right):
        return ((left & right) if maximum else (left | right)) & 0x80000000
    if eq_word(left,right): return left
    ll = less_word(left,right)
    return right if maximum and ll else left if maximum else left if ll else right

def unique_words():
    seen = set(); result = []
    for left,right in zip(PAIR_LEFT,PAIR_RIGHT):
        for word in (left,right):
            if word not in seen:
                seen.add(word); result.append(word)
    return result

UNIQUE = unique_words()

def broadcast_pairs():
    result=[]
    for i0 in range(2):
        for i1 in range(4):
            for i2 in range(3):
                result.append((BROADCAST_LEFT[i0*3+i2], BROADCAST_RIGHT[i1*3+i2]))
    return result

def parse_raw(path: Path):
    meta={}; outputs={}
    raw=path.read_bytes(); text=raw.decode("utf-8")
    for line in text.splitlines():
        if line.startswith(("OUT8 ","OUT32 ","MID32 ")):
            parts=line.split()
            role=parts[0]
            fields=dict(part.split("=",1) for part in parts[1:])
            vals=[int(v,16) for v in fields["values"].split(",") if v]
            if len(vals)!=int(fields["count"]): raise AssertionError(f"count mismatch {fields['name']}")
            outputs[fields["name"]]={"role":role,"shape":fields["shape"],"values":vals,"canary":fields["canary"]}
        elif "=" in line:
            key,value=line.split("=",1); meta[key]=value
    return raw,meta,outputs

def accept_expected(actual: int, expected) -> bool:
    return is_nan(actual) if expected == NAN else actual == expected

def require_values(outputs, name, expected, failures):
    row=outputs.get(name)
    if row is None:
        failures.append(f"missing {name}"); return
    if row["canary"]!="PASS": failures.append(f"canary {name}")
    if len(row["values"])!=len(expected): failures.append(f"length {name}"); return
    for index,(actual,want) in enumerate(zip(row["values"],expected)):
        if not accept_expected(actual,want):
            failures.append(f"{name}[{index}] actual={actual:08x} expected={want}")
            return

def finite_fraction(word: int) -> Fraction:
    sign=-1 if word & 0x80000000 else 1
    exponent=(word>>23)&0xff; fraction=word&0x7fffff
    if exponent==0:
        return Fraction(sign*fraction, 1<<149)
    mantissa=(1<<23)|fraction
    power=exponent-127-23
    return Fraction(sign*mantissa*(1<<power),1) if power>=0 else Fraction(sign*mantissa,1<<(-power))

def floor_log2(value: Fraction) -> int:
    n=value.numerator; d=value.denominator
    e=n.bit_length()-d.bit_length()
    if e>=0:
        if Fraction(1<<e,1)>value: e-=1
    else:
        if Fraction(1,1<<(-e))>value: e-=1
    return e

def round_even(value: Fraction) -> int:
    q,r=divmod(value.numerator,value.denominator)
    twice=r*2
    if twice>value.denominator or (twice==value.denominator and (q&1)): q+=1
    return q

def round_float32(value: Fraction, zero_sign: int = 0) -> int:
    if value==0: return NEG_ZERO if zero_sign else POS_ZERO
    sign=1 if value<0 else 0; magnitude=abs(value)
    overflow=Fraction((1<<128)-(1<<103),1)
    if magnitude>=overflow: return (sign<<31)|POS_INF
    e=floor_log2(magnitude)
    if e>=-126:
        scale=e-23
        scaled=magnitude/Fraction(1<<scale,1) if scale>=0 else magnitude*Fraction(1<<(-scale),1)
        mantissa=round_even(scaled)
        if mantissa==(1<<24):
            mantissa>>=1; e+=1
        if e>127: return (sign<<31)|POS_INF
        return (sign<<31)|((e+127)<<23)|(mantissa-(1<<23))
    scaled=magnitude*Fraction(1<<149,1)
    mantissa=round_even(scaled)
    if mantissa==0: return NEG_ZERO if sign else POS_ZERO
    if mantissa>=(1<<23): return (sign<<31)|(1<<23)
    return (sign<<31)|mantissa

def operand_alternatives(value):
    if value==NAN: return (NAN,)
    if is_subnormal(value): return (value, value&0x80000000)
    return (value,)

def result_alternatives(word):
    if word==NAN: return {NAN}
    if is_subnormal(word): return {word,POS_ZERO,NEG_ZERO}
    return {word}

def primitive_no_daz(left, right, product: bool):
    if left==NAN or right==NAN: return NAN
    if is_inf(left) or is_inf(right):
        if product:
            if (is_inf(left) and is_zero(right)) or (is_inf(right) and is_zero(left)): return NAN
            sign=((left^right)>>31)&1
            return NEG_INF if sign else POS_INF
        if is_inf(left) and is_inf(right) and ((left^right)&0x80000000): return NAN
        return left if is_inf(left) else right
    if product:
        if is_zero(left) or is_zero(right): return NEG_ZERO if ((left^right)&0x80000000) else POS_ZERO
        return round_float32(finite_fraction(left)*finite_fraction(right))
    value=finite_fraction(left)+finite_fraction(right)
    if value==0:
        both_negative_zero=is_zero(left) and is_zero(right) and (left&0x80000000) and (right&0x80000000)
        return NEG_ZERO if both_negative_zero else POS_ZERO
    return round_float32(value)

def primitive_results(left,right,product):
    result=set()
    for a in operand_alternatives(left):
        for b in operand_alternatives(right):
            result.update(result_alternatives(primitive_no_daz(a,b,product)))
    return frozenset(result)

@lru_cache(maxsize=None)
def tree_results(sequence, product):
    if len(sequence)==1:
        value=sequence[0]
        return frozenset({NAN if is_nan(value) else value})
    result=set()
    for split in range(1,len(sequence)):
        for left in tree_results(sequence[:split],product):
            for right in tree_results(sequence[split:],product):
                result.update(primitive_results(left,right,product))
    return frozenset(result)

def scan_expected(words,dims,axis,exclusive,reverse,product):
    axis_len=dims[axis]
    inner=1
    for dim in dims[axis+1:]: inner*=dim
    lines=len(words)//axis_len
    actual_allowed=[None]*len(words); max_set=1
    identity=0x3f800000 if product else 0x00000000
    for line in range(lines):
        outer=line//inner; inner_index=line%inner
        base=outer*axis_len*inner+inner_index
        order=list(range(axis_len))
        if reverse: order.reverse()
        sequence=[]
        for logical in order:
            offset=base+logical*inner
            prefix=tuple(sequence) if exclusive else tuple(sequence+[words[offset]])
            allowed=frozenset({identity}) if not prefix else tree_results(prefix,product)
            actual_allowed[offset]=allowed; max_set=max(max_set,len(allowed))
            sequence.append(words[offset])
    return actual_allowed,max_set

def validate(raw_path: Path):
    raw,meta,outputs=parse_raw(raw_path)
    failures=[]; candidate_failures={name:[] for name in CUSTOM_CANDIDATES}; max_sets={}
    for key,want in {"TASK":"0052","DEVICE_CONTEXTS":"1","COMMAND_BUFFERS":"1","FAST_MATH":"0",
                     "COMPILED_PIPELINES":"15","INPUT_COUNT":"16","OUTPUT_COUNT":"98",
                     "INPUTS_PRESERVED":"PASS","ORACLE_PROCESS":"PASS"}.items():
        if meta.get(key)!=want: failures.append(f"metadata {key}={meta.get(key)} expected={want}")
    bcast=broadcast_pairs()
    for op in ("gt","ge","lt","le","eq","ne"):
        local=candidate_failures[f"custom_cmp_{op}"]
        require_values(outputs,f"cmp_{op}_pairs",[compare_word(a,b,op) for a,b in zip(PAIR_LEFT,PAIR_RIGHT)],local)
        require_values(outputs,f"cmp_{op}_broadcast",[compare_word(a,b,op) for a,b in bcast],local)
    for label,maximum in (("min",False),("max",True)):
        local=candidate_failures[f"custom_tensor_{label}"]
        require_values(outputs,f"tensor_{label}_pairs",[extreme_word(a,b,maximum) for a,b in zip(PAIR_LEFT,PAIR_RIGHT)],local)
        require_values(outputs,f"tensor_{label}_broadcast",[extreme_word(a,b,maximum) for a,b in bcast],local)
        local=candidate_failures[f"custom_scalar_{label}"]
        for scalar in SCALARS:
            require_values(outputs,f"scalar_{label}_s{scalar:08x}",[extreme_word(x,scalar,maximum) for x in UNIQUE],local)
    for lower,upper in CLAMPS:
        suffix=f"_lo{lower:08x}_hi{upper:08x}"
        middle=[extreme_word(x,lower,True) for x in UNIQUE]
        final=[NAN if m==NAN else extreme_word(m,upper,False) for m in middle]
        require_values(outputs,"clamp_fused"+suffix,final,candidate_failures["custom_clamp_fused"])
        require_values(outputs,"clamp_composed_middle"+suffix,middle,candidate_failures["composed_custom_clamp"])
        require_values(outputs,"clamp_composed"+suffix,final,candidate_failures["composed_custom_clamp"])
    dims=[2,3,8]
    forms={"full":0x7,"axis1_drop":0x2,"axis1_keep":0x2,"axes20_drop":0x5,"axes20_keep":0x5,"identity":0}
    def reduce_expected(words,dims,mask,maximum):
        rank=len(dims); reduced=[i for i in range(rank) if mask&(1<<i)]; kept=[i for i in range(rank) if not mask&(1<<i)]
        out_count=1
        for i in kept: out_count*=dims[i]
        red_count=1
        for i in reduced: red_count*=dims[i]
        strides=[1]*rank
        for i in range(rank-2,-1,-1): strides[i]=strides[i+1]*dims[i+1]
        result=[]
        for out in range(out_count):
            rem=out; coords=[0]*rank
            for i in reversed(kept): coords[i]=rem%dims[i]; rem//=dims[i]
            vals=[]
            for red in range(red_count):
                rr=red; c=coords[:]
                for i in reversed(reduced): c[i]=rr%dims[i]; rr//=dims[i]
                vals.append(words[sum(c[i]*strides[i] for i in range(rank))])
            acc=vals[0]
            for value in vals[1:]:
                if acc==NAN: break
                acc=extreme_word(acc,value,maximum)
            result.append(acc)
        return result
    for label,maximum in (("min",False),("max",True)):
        local=candidate_failures[f"custom_reduction_{label}"]
        for form,mask in forms.items():
            require_values(outputs,f"reduction_{label}_base_{form}",reduce_expected(AGGREGATE,dims,mask,maximum),local)
        for edge,words in REDUCTION_EDGES.items():
            require_values(outputs,f"reduction_{label}_{edge}",reduce_expected(words,[len(words)],1,maximum),local)
    for label,product in (("sum",False),("prod",True)):
        local=candidate_failures[f"custom_scan_{label}"]
        scan_forms=[("axis2_inc_fwd",2,False,False),("axis2_exc_fwd",2,True,False),
                    ("axis2_inc_rev",2,False,True),("axis2_exc_rev",2,True,True),
                    ("axis1_inc_fwd",1,False,False)]
        local_max=1
        for name,axis,exclusive,reverse in scan_forms:
            allowed,max_set=scan_expected(AGGREGATE,dims,axis,exclusive,reverse,product)
            local_max=max(local_max,max_set)
            row=outputs.get(f"scan_{label}_{name}")
            if row is None:
                local.append(f"missing scan_{label}_{name}"); continue
            if row["canary"]!="PASS": local.append(f"canary scan_{label}_{name}")
            for index,(actual,choices) in enumerate(zip(row["values"],allowed)):
                accepted=(is_nan(actual) if NAN in choices else False) or actual in choices
                if not accepted:
                    rendered=sorted(f"{x:08x}" for x in choices if x!=NAN)
                    if NAN in choices: rendered.append("NAN")
                    local.append(f"scan_{label}_{name}[{index}]={actual:08x} allowed={rendered}")
                    break
        max_sets[f"custom_scan_{label}"]=local_max
    for name,items in candidate_failures.items(): failures.extend(f"{name}: {item}" for item in items)
    corpus={"pairs":list(zip(PAIR_LEFT,PAIR_RIGHT)),"broadcast_left":BROADCAST_LEFT,
            "broadcast_right":BROADCAST_RIGHT,"scalars":SCALARS,"clamps":CLAMPS,
            "aggregate":AGGREGATE,"reduction_edges":REDUCTION_EDGES,
            "scan_forms":["axis2_inc_fwd","axis2_exc_fwd","axis2_inc_rev","axis2_exc_rev","axis1_inc_fwd"]}
    corpus_hash=hashlib.sha256(json.dumps(corpus,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    raw_hash=hashlib.sha256(raw).hexdigest()
    print("VALIDATOR=task0052-exact-v1")
    print(f"CORPUS_SHA256={corpus_hash}")
    print(f"RAW_OUTPUT_SHA256={raw_hash}")
    print(f"RAW_OUTPUT_COUNT={len(outputs)}")
    for candidate in DIRECT_CANDIDATES:
        print(f"CANDIDATE={candidate} API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract")
    for candidate in CUSTOM_CANDIDATES:
        verdict="FAIL" if candidate_failures[candidate] else "PASS"
        related=[]
        if candidate.startswith("custom_cmp_"):
            stem=candidate.replace("custom_",""); related=[n for n in outputs if n.startswith(stem)]
        elif candidate.startswith("custom_tensor_"):
            stem=candidate.replace("custom_",""); related=[n for n in outputs if n.startswith(stem)]
        elif candidate.startswith("custom_scalar_"):
            stem=candidate.replace("custom_",""); related=[n for n in outputs if n.startswith(stem)]
        elif candidate=="custom_clamp_fused": related=[n for n in outputs if n.startswith("clamp_fused")]
        elif candidate=="composed_custom_clamp": related=[n for n in outputs if n.startswith("clamp_composed")]
        elif candidate.startswith("custom_reduction_"):
            stem=candidate.replace("custom_",""); related=[n for n in outputs if n.startswith(stem)]
        elif candidate.startswith("custom_scan_"):
            stem=candidate.replace("custom_",""); related=[n for n in outputs if n.startswith(stem)]
        digest=hashlib.sha256()
        for name in sorted(related):
            digest.update(name.encode()); digest.update(b"\0")
            width=1 if outputs[name]["role"]=="OUT8" else 4
            for value in outputs[name]["values"]: digest.update(value.to_bytes(width,"little"))
        extra=f" MAX_REACHABLE_SET={max_sets[candidate]}" if candidate in max_sets else ""
        print(f"CANDIDATE={candidate} API=PASS DOMAIN=PASS NUMERICAL={verdict} OUTPUT_SHA256={digest.hexdigest()} CASE_OUTPUTS={len(related)}{extra} FAILURES={len(candidate_failures[candidate])}")
    if failures:
        for item in failures[:50]: print("FAIL="+item)
        print(f"ORACLE_VERDICT=FAIL FAILURES={len(failures)}")
        return 1
    print("ORACLE_VERDICT=PASS FAILURES=0")
    return 0

if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("usage: validate.py RAW_OUTPUT")
    raise SystemExit(validate(Path(sys.argv[1])))
