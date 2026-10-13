export const roundHalfAway = (x, digits) => {
  const [int, frac] = Math.abs(x).toFixed(9).split(".");
  const unit = 10n ** BigInt(9 - digits);
  const q = (BigInt(int + frac) + unit / 2n) / unit;
  const s = q.toString().padStart(digits + 1, "0");
  return (x < 0 && q !== 0n ? "-" : "") + (digits ? `${s.slice(0, -digits)}.${s.slice(-digits)}` : s);
};

export const halfWidth = (est, [lo, hi], digits, scale = 1) => roundHalfAway(scale * Math.max(est - lo, hi - est), digits);
