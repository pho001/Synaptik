import Task0069Binary32

namespace Task0070Sign

open Task0069

def rawBits (word : Word) : Nat := (Task0069.rawWord word).val

/-- Literal raw-u32 semantics of the frozen MSL `sign_bits` helper. -/
def signBitsSource (word : Word) : Nat :=
  if rawClass word = .nan ∨ rawClass word = .zero then rawBits word
  else if word.sign then 0xbf800000 else 0x3f800000

/-- Result-set model for SIGN, including payload-preserving NaN and signed zero. -/
def SignModel (word : Word) (result : Nat) : Prop :=
  if rawClass word = .nan ∨ rawClass word = .zero then
    result = rawBits word
  else
    result = if word.sign then 0xbf800000 else 0x3f800000

theorem sign_universal_raw_certificate (word : Word) :
    SignModel word (signBitsSource word) := by
  unfold SignModel signBitsSource
  by_cases preserved : rawClass word = .nan ∨ rawClass word = .zero
  · simp [preserved]
  · simp [preserved]

theorem sign_permitted_iff (word : Word) (result : Nat) :
    SignModel word result ↔ result = signBitsSource word := by
  unfold SignModel signBitsSource
  by_cases preserved : rawClass word = .nan ∨ rawClass word = .zero
  · simp [preserved]
  · simp [preserved]

theorem sign_all_raw32_certificate (raw : Raw32) :
    SignModel (wordOfRaw raw) (signBitsSource (wordOfRaw raw)) :=
  sign_universal_raw_certificate (wordOfRaw raw)

/-- NaN payload/sign are returned bit-for-bit. -/
theorem sign_nan_raw_identity (word : Word) (nan : rawClass word = .nan) :
    signBitsSource word = rawBits word := by
  simp [signBitsSource, nan]

/-- Both signed zeros are returned bit-for-bit. -/
theorem sign_zero_raw_identity (word : Word) (zero : rawClass word = .zero) :
    signBitsSource word = rawBits word := by
  simp [signBitsSource, zero]

end Task0070Sign
