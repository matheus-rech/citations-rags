from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd


def to_rows(extraction: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta = extraction.get('study_metadata', {})
    study_id = f"{meta.get('first_author','unknown')}_{meta.get('year','0000')}"
    classification = extraction.get('classification')

    # Arms table rows (optional)
    for arm in extraction.get('arms', []) or []:
        rows.append({
            'study_id': study_id,
            'table': 'arms',
            'classification': classification,
            **{k: arm.get(k) for k in ['arm_id','label','type','n','notes']}
        })

    # Outcomes rows
    for o in extraction.get('outcomes', []) or []:
        rows.append({
            'study_id': study_id,
            'table': 'outcomes',
            'classification': classification,
            'name': o.get('name'),
            'type': o.get('type'),
            'definition': o.get('definition'),
            'scale': o.get('scale'),
            'timepoint': o.get('timepoint'),
            'arm_id': o.get('arm_id'),
            'comparison': o.get('comparison'),
            'statistic_type': o.get('statistic_type'),
            'value': o.get('value'),
            'dispersion_type': o.get('dispersion_type'),
            'dispersion_value': o.get('dispersion_value'),
            'n': o.get('n'),
            'subgroup_label': o.get('subgroup_label'),
            'sensitivity_label': o.get('sensitivity_label'),
            'provenance_source': (o.get('provenance') or {}).get('source'),
            'provenance_offset_status': (o.get('provenance') or {}).get('offset_status'),
            'provenance_start_offset': (o.get('provenance') or {}).get('start_offset'),
            'provenance_end_offset': (o.get('provenance') or {}).get('end_offset'),
            'provenance_page': (o.get('provenance') or {}).get('page'),
        })

    return rows


def harmonize_to_csv(extractions: List[Dict[str, Any]], out_path: str) -> None:
    all_rows: List[Dict[str, Any]] = []
    for ex in extractions:
        all_rows.extend(to_rows(ex))
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
