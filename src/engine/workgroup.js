export const WORKGROUPS = [[8, 8], [16, 8], [16, 16]];
export function validateWorkgroup(value = [8, 8], limits = {}) {
  if (!WORKGROUPS.some(([x, y]) => value?.[0] === x && value?.[1] === y && value.length === 2)) {
    throw new Error('Workgroup não suportado');
  }
  const [x, y] = value;
  if (x > (limits.maxComputeWorkgroupSizeX ?? 256) || y > (limits.maxComputeWorkgroupSizeY ?? 256)
    || x * y > (limits.maxComputeInvocationsPerWorkgroup ?? 256)) throw new Error('Workgroup excede o limite da GPU');
  return [x, y];
}
export function workgroupShader(shader, group) {
  const [x, y] = validateWorkgroup(group);
  return shader.replace('@workgroup_size(8, 8)', `@workgroup_size(${x}, ${y})`);
}
