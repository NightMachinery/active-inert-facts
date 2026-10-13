export function figureLabels(ids, { labelOf, sourceOf, setName, sources }) {
  const names = new Set(sources.map((s) => setName(s)));
  const bare = new Map();
  for (const id of ids) {
    const label = labelOf.get(id);
    if (!label) throw new Error(`relation ${id} has no label`);
    const m = label.match(/^(.*) \(([^()]+)\)$/);
    const own = setName(sourceOf.get(id));
    if (m && m[2] !== own && names.has(m[2])) throw new Error(`relation ${id}: label "${label}" carries another set's name`);
    bare.set(id, m && m[2] === own ? m[1] : label);
  }
  const count = new Map();
  for (const b of bare.values()) count.set(b, (count.get(b) || 0) + 1);
  const out = new Map();
  for (const id of ids) {
    const b = bare.get(id);
    out.set(id, count.get(b) > 1 ? `${b} (${setName(sourceOf.get(id))})` : b);
  }
  if (new Set(out.values()).size !== out.size) throw new Error("two relations still share a figure label after disambiguation");
  return out;
}
