export const HEADER_BYTES = 24;
export const MAGIC = 0x31435341;
export function repeatResponse(buffer,id,width,height) {
  if(buffer.byteLength<4 || new DataView(buffer).getUint32(0,true)!==0x34435341)return false;
  if(buffer.byteLength!==HEADER_BYTES)throw new Error('Resposta repetida com tamanho inválido.');
  const view=new DataView(buffer);
  if([0x34435341,id,width,height,width,height].some((v,i)=>view.getUint32(i*4,true)!==v))
    throw new Error('Resposta repetida não corresponde ao quadro.');
  return true;
}
export function rgbResponse(buffer,id,width,height) {
  if(buffer.byteLength<HEADER_BYTES)return null;
  const view=new DataView(buffer);
  if(view.getUint32(0,true)!==0x33435341)return null;
  if(buffer.byteLength!==HEADER_BYTES+width*height*3)throw new Error('RGB local com tamanho inválido.');
  if([0x33435341,id,width,height,width,height].some((v,i)=>view.getUint32(i*4,true)!==v))throw new Error('RGB local não corresponde ao quadro.');
  return new Uint8Array(buffer,HEADER_BYTES);
}
export function jpegResponse(buffer,id,width,height) {
  if(buffer.byteLength<HEADER_BYTES+4)return null;
  const view=new DataView(buffer);
  if(view.getUint32(0,true)!==0x32435341)return null;
  if(buffer.byteLength>HEADER_BYTES+width*height*4+65536)throw new Error('JPEG local excedeu o limite.');
  const expected=[0x32435341,id,width,height,width,height];
  if(expected.some((value,index)=>view.getUint32(index*4,true)!==value))throw new Error('JPEG local não corresponde ao quadro.');
  const bytes=new Uint8Array(buffer,HEADER_BYTES);
  if(bytes[0]!==255||bytes[1]!==216||bytes.at(-2)!==255||bytes.at(-1)!==217)throw new Error('JPEG local inválido.');
  return new Blob([bytes],{type:'image/jpeg'});
}
export function writeHeader(buffer, id, width, height, outputWidth, outputHeight) {
  const view = new DataView(buffer);
  [MAGIC, id, width, height, outputWidth, outputHeight].forEach((value, index) => view.setUint32(index * 4, value, true));
}
export function readResponse(buffer, id, width, height) {
  if (buffer.byteLength !== HEADER_BYTES + width * height * 4) throw new Error('Resposta ROCm com tamanho inválido.');
  const view = new DataView(buffer);
  const expected = [MAGIC, id, width, height, width, height];
  if (expected.some((value, index) => view.getUint32(index * 4, true) !== value)) throw new Error('Resposta ROCm não corresponde ao quadro enviado.');
  return new Uint8ClampedArray(buffer, HEADER_BYTES);
}
