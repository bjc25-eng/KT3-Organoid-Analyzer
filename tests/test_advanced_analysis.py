from pathlib import Path

import numpy as np
import pandas as pd

import advanced_analysis as aa


def synthetic_tracking():
    rows = []
    # Four concentrations, three tracked wells each, three timepoints.
    conditions = [
        (1, 'Vehicle', 0.0, True),
        (2, 'Dose 1', 10.0, True),
        (3, 'Dose 2', 100.0, True),
        (4, 'Dose 3', 1000.0, True),
    ]
    times = [(1, 'Day 0', 0.0), (2, 'Day 3', 3.0), (3, 'Day 7', 7.0)]
    final_multipliers = {1: 2.2, 2: 1.5, 3: 0.8, 4: 0.35}
    for ci, cname, conc, rfp in conditions:
        for well_n in range(1, 4):
            baseline = 1000.0 + 100.0 * well_n
            for ti, tname, elapsed in times:
                if ti == 1:
                    area = baseline
                elif ti == 2:
                    area = baseline * (1.0 + (final_multipliers[ci]-1.0)*0.45)
                else:
                    area = baseline * final_multipliers[ci]
                rows.append({
                    'condition_index': ci,
                    'condition': cname,
                    'timepoint_index': ti,
                    'timepoint': tname,
                    'elapsed_time': elapsed,
                    'concentration': conc,
                    'well_index': f'{well_n},1',
                    'PDO_count': 1,
                    'PSC_like_focus_count': float(well_n + ci),
                    'RFP_PSC_stromal_cells_present': rfp,
                    'um_per_pixel': 1.0,
                    'total_PDO_projected_area_um2': area,
                    'mean_PDO_diameter_um': np.sqrt(area),
                    'max_PDO_diameter_um': np.sqrt(area),
                    'PDO_present': True,
                })
    return pd.DataFrame(rows)


def test_add_response_metrics_baseline_is_one():
    d = aa.add_response_metrics(synthetic_tracking())
    baseline = d[d.timepoint_index == 1]
    assert np.allclose(baseline.relative_total_PDO_area_vs_baseline, 1.0)
    assert np.allclose(baseline.percent_area_change_vs_baseline, 0.0)


def test_final_response_table_has_one_row_per_condition_well():
    final = aa.final_response_table(synthetic_tracking())
    assert len(final) == 12
    assert (final.timepoint_index == 3).all()


def test_waterfall_outputs_csv_and_figures(tmp_path):
    result = aa.analyse_waterfall(synthetic_tracking(), tmp_path)
    assert result['status'] == 'complete'
    assert (tmp_path/'waterfall_final_well_responses.csv').exists()
    assert len(list(tmp_path.glob('waterfall_*.png'))) == 4


def test_heatmap_outputs_matrix_per_condition(tmp_path):
    result = aa.analyse_heatmaps(synthetic_tracking(), tmp_path)
    assert result['status'] == 'complete'
    assert len(list(tmp_path.glob('heatmap_*.png'))) == 4
    assert len(list(tmp_path.glob('heatmap_matrix_*.csv'))) == 4


def test_distribution_outputs_ecdfs(tmp_path):
    result = aa.analyse_distributions(synthetic_tracking(), tmp_path)
    assert result['status'] == 'complete'
    assert (tmp_path/'final_response_violin_by_condition.png').exists()
    assert len(list(tmp_path.glob('ECDF_time_*.png'))) >= 1


def test_response_classification_uses_supplied_thresholds(tmp_path):
    result = aa.analyse_responder_classes(synthetic_tracking(), tmp_path, -20.0, 20.0)
    assert result['status'] == 'complete'
    df = pd.read_csv(tmp_path/'well_response_classification.csv')
    assert set(df.response_class.unique()).issubset({'Responder', 'Intermediate', 'Resistant'})
    assert (df.responder_max_pct_threshold == -20.0).all()
    assert (df.resistant_min_pct_threshold == 20.0).all()


def test_invalid_classification_thresholds_raise(tmp_path):
    try:
        aa.analyse_responder_classes(synthetic_tracking(), tmp_path, 30.0, 20.0)
    except ValueError as exc:
        assert 'lower' in str(exc)
    else:
        raise AssertionError('Expected ValueError')


def test_psc_association_outputs_stats(tmp_path):
    result = aa.analyse_psc_association(synthetic_tracking(), tmp_path)
    assert result['status'] == 'complete'
    stats = pd.read_csv(tmp_path/'PSC_response_spearman_by_condition_time.csv')
    assert 'spearman_rho' in stats.columns
    assert 'p_value_BH_FDR' in stats.columns


def test_growth_and_dose_outputs_4pl_files(tmp_path):
    result = aa.analyse_growth_and_dose(synthetic_tracking(), tmp_path, 'nM', 'days')
    assert result['status'] == 'complete'
    assert (tmp_path/'growth_trajectory_by_condition.png').exists()
    assert (tmp_path/'dose_response_summary.csv').exists()
    assert (tmp_path/'dose_response_4PL_fit.csv').exists()
    fit = pd.read_csv(tmp_path/'dose_response_4PL_fit.csv')
    assert len(fit) == 1


def test_growth_without_concentration_skips_only_dose_fit(tmp_path):
    d = synthetic_tracking().drop(columns=['concentration'])
    result = aa.analyse_growth_and_dose(d, tmp_path, 'nM', 'days')
    assert result['status'] == 'partial'
    assert (tmp_path/'growth_trajectory_by_condition.png').exists()
    assert not (tmp_path/'dose_response_summary.csv').exists()


def test_run_selected_analyses_only_creates_selected_modules(tmp_path):
    selected = [aa.ANALYSIS_WATERFALL, aa.ANALYSIS_CLASSIFICATION]
    manifest = aa.run_selected_analyses(
        synthetic_tracking(), tmp_path, selected,
        responder_max_pct=-20.0, resistant_min_pct=20.0
    )
    assert set(manifest.analysis) == set(selected)
    assert (tmp_path/'Individual_PDO_growth_waterfall').exists()
    assert (tmp_path/'Resistant_responding_population_analysis').exists()
    assert not (tmp_path/'Well_time_response_heatmap').exists()
