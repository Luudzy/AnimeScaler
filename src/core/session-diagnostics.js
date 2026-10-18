const STALE_ORIGINS = new Set(['inline_engine', 'native_client', 'native_host', 'background_relay']);
const CONTEXT_KEYS = new Set(['messageType', 'sequence', 'expectedSequence', 'receivedSequence',
  'revision', 'expectedRevision', 'generation', 'closed', 'hasPending', 'hasInFlight']);

// FNV-1a is a correlation tag, not a security hash. Protocol IDs remain out of logs.
export function sessionTag(value) {
  if (typeof value !== 'string' || value.length === 0 || value.length > 256) return null;
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

export function isStaleOrigin(value) {
  return STALE_ORIGINS.has(value);
}

export function sessionDiagnostic(origin, id, details = {}) {
  const result = {};
  if (isStaleOrigin(origin)) result.staleOrigin = origin;
  const tag = sessionTag(id);
  if (tag) result.sessionTag = tag;
  for (const [key, value] of Object.entries(details)) {
    if (!CONTEXT_KEYS.has(key)) continue;
    if (typeof value === 'boolean' || (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0)) {
      result[key] = value;
    } else if (key === 'messageType' && typeof value === 'string' && /^[A-Z_]{1,32}$/.test(value)) {
      result[key] = value;
    }
  }
  return result;
}
