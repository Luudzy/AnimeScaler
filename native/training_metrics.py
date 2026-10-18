"""Independent quality comparisons; development gates never inspect test scores."""
def comparisons(scores):
    candidate, baseline, bilinear = (scores[k] for k in ('candidate','previous','bilinear'))
    result = {}
    for field, label in [('mae','Mae'),('edgeMae','Edge')]:
        for reference, name in [(bilinear,'Bilinear'),(baseline,'Baseline')]:
            result[f'beats{name}{label}'] = candidate[field] < reference[field]
            result[f'{label[0].lower()+label[1:]}Vs{name}Percent'] = 100*(candidate[field]/max(reference[field],1e-12)-1)
    return result

def quality_report(metrics):
    reports = {split:comparisons(scores) for split,scores in metrics.items()}
    validation = reports['validation']
    gate = all(validation[key] for key in ('beatsBilinearMae','beatsBilinearEdge','beatsBaselineMae','beatsBaselineEdge'))
    return {'comparisons':reports,'passesValidationGate':gate,'qualityGatePassed':gate,
            'improvesOverBilinear':validation['beatsBilinearMae'] and validation['beatsBilinearEdge'],
            'selectionSplit':'validation'}
