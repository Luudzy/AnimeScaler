export function setupDiagnostics() {
  const status = document.getElementById('diagnosticStatus');
  const exportButton = document.getElementById('exportDiagnostics');
  const clearButton = document.getElementById('clearDiagnostics');
  const request = async type => {
    const reply = await chrome.runtime.sendMessage({type});
    if (!reply?.ok) throw new Error(reply?.error ?? 'Não foi possível acessar o histórico.');
    return reply.data;
  };
  exportButton.addEventListener('click', async () => {
    exportButton.disabled = true;
    let url;
    try {
      const history = await request('ANIMESCALER_DIAGNOSTIC_EXPORT');
      const report = {...history, exportedAt: new Date().toISOString(), version: chrome.runtime.getManifest().version,
        notes: 'Resumos de 5 segundos; FPS inclui pausas/esperas. sessionRunMeanMs mede a chamada JS, não tempo GPU por operador. Sem frames ou URLs. Últimos segundos podem ainda estar em memória.'};
      url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `animescaler-diagnostics-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
      anchor.click();
      status.textContent = `${history.events.length} registros exportados. Guarde o arquivo em Downloads para análise depois.`;
    } catch (error) { status.textContent = error.message; }
    finally {
      if (url !== undefined) setTimeout(() => URL.revokeObjectURL(url), 10000);
      exportButton.disabled = false;
    }
  });
  clearButton.addEventListener('click', async () => {
    try {
      await request('ANIMESCALER_DIAGNOSTIC_CLEAR');
      status.textContent = 'Histórico apagado. Novos registros serão salvos durante o uso.';
    } catch (error) { status.textContent = error.message; }
  });
}
