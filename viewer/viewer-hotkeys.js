export function handleViewerKeydown(event, state, actions) {
  if (event.repeat || event.ctrlKey || event.altKey || event.metaKey
      || event.target.isContentEditable || event.target.matches('input,select,textarea')) return false;

  const comparisonControl = ['Space', 'ArrowLeft', 'ArrowRight'].includes(event.code);
  if (state.comparing && comparisonControl) {
    event.preventDefault();
    return true;
  }

  if (event.code === 'Escape') {
    event.preventDefault();
    if (state.comparing && !state.comparisonBusy) actions.closeComparison();
    else actions.exitFullscreen();
    return true;
  }
  if (event.code === 'KeyF') {
    event.preventDefault();
    actions.toggleFullscreen();
  } else if (event.code === 'KeyI') {
    event.preventDefault();
    actions.toggleInfo();
  } else if (event.code === 'KeyC') {
    event.preventDefault();
    actions.toggleComparison();
  } else if (event.code === 'Space') {
    if (event.target.closest('button')) return false;
    event.preventDefault();
    actions.togglePlayback();
  } else if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
    event.preventDefault();
    actions.seek(event.code === 'ArrowLeft' ? -10 : 10);
  } else {
    return false;
  }
  return true;
}
