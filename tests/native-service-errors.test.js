import test from 'node:test';
import assert from 'node:assert/strict';
import { serviceErrorText, classifyNativeServiceError, classifyNativeServiceClose } from '../src/engine/native-service-contract.js';

test('D3D12 transport errors identify FSR 3 instead of ROCm', () => {
  assert.equal(serviceErrorText('d3d12', 'timeout'), 'Serviço FSR 3 excedeu o tempo de resposta.');
  assert.equal(serviceErrorText('d3d12', 'disconnected'), 'Serviço FSR 3 desconectado. Abra Iniciar-FSR3.cmd.');
  assert.equal(serviceErrorText('d3d12', 'invalid-response'), 'Resposta FSR 3 inválida.');
});

test('ROCm transport errors keep their backend-specific guidance', () => {
  assert.equal(serviceErrorText('rocm', 'timeout'), 'Serviço ROCm excedeu o tempo de resposta.');
  assert.equal(serviceErrorText('rocm', 'disconnected'), 'Serviço ROCm desconectado. Abra Iniciar-ROCm.cmd.');
});

test('native service errors expose stable codes and never forward arbitrary server text', () => {
  assert.deepEqual(classifyNativeServiceError('d3d12', 'GPU ocupada.'), {
    code: 'NATIVE_SESSION_BUSY', message: 'FSR 3 está ocupado por outra sessão.',
  });
  const privateError = classifyNativeServiceError('rocm', 'C:\\Users\\private\\token=secret');
  assert.deepEqual(privateError, { code: 'BACKEND_UNAVAILABLE', message: 'Falha no serviço ROCm.' });
  assert.equal(JSON.stringify(privateError).includes('secret'), false);
});

test('native service close reasons map to stable backend-aware errors', () => {
  assert.deepEqual(classifyNativeServiceClose('d3d12', { code: 1013, reason: 'FSR3 viewer already active' }), {
    code: 'NATIVE_SESSION_BUSY', message: 'FSR 3 está ocupado por outra sessão.',
  });
  assert.deepEqual(classifyNativeServiceClose('d3d12', { code: 1011, reason: 'private traceback' }), {
    code: 'BACKEND_UNAVAILABLE', message: 'Serviço FSR 3 desconectado. Abra Iniciar-FSR3.cmd.',
  });
});

test('TensorRT and OpenCL failures do not display ROCm guidance', () => {
  assert.equal(serviceErrorText('tensorrt', 'disconnected'),
    'Serviço TensorRT desconectado. Abra Iniciar-TensorRT.cmd.');
  assert.equal(serviceErrorText('rocm', 'disconnected', 'anime4kcpp-acnet-f8b4'),
    'Worker OpenCL desconectado. Verifique o servi\u00e7o nativo configurado.');
  assert.equal(serviceErrorText('opencl','disconnected','anime4kcpp-acnet-f8b4'),
    'Servi\u00e7o OpenCL desconectado. Abra Iniciar-OpenCL.cmd.');
  assert.equal(serviceErrorText('opencl','disconnected'),
    'Servi\u00e7o OpenCL desconectado. Abra Iniciar-OpenCL.cmd.');
});
