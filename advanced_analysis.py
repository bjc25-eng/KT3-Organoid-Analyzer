from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

ANALYSIS_GROWTH = 'Dose-response / growth-rate curve'
ANALYSIS_WATERFALL = 'Individual PDO growth waterfall'
ANALYSIS_HEATMAP = 'Well × time response heatmap'
ANALYSIS_DISTRIBUTIONS = 'Full response distributions for every condition'
ANALYSIS_CLASSIFICATION = 'Resistant/responding population analysis'
ANALYSIS_PSC = 'PSC-associated drug-response analysis'

ANALYSIS_OPTIONS = [
    ANALYSIS_GROWTH,
    ANALYSIS_WATERFALL,
    ANALYSIS_HEATMAP,
    ANALYSIS_DISTRIBUTIONS,
    ANALYSIS_CLASSIFICATION,
    ANALYSIS_PSC,
]


def _safe_name(text: str) -> str:
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text).strip()).strip('_')
    return value or 'condition'


def add_response_metrics(ldf: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, baseline-normalised longitudinal table.

    The function is deliberately idempotent: it removes any previously derived
    response columns before recalculating them. This prevents duplicate _x/_y
    baseline columns when one analysis module calls another.
    """
    if ldf is None or not len(ldf):
        return pd.DataFrame() if ldf is None else ldf.copy()

    out = ldf.copy()
    derived_prefixes = (
        'baseline_elapsed_time',
        'baseline_total_PDO_area_um2_calc',
        'relative_total_PDO_area_vs_baseline',
        'percent_area_change_vs_baseline',
        'log_area_growth_rate_per_time',
    )
    drop_cols = [c for c in out.columns if any(c == p or c.startswith(p + '_') for p in derived_prefixes)]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    required = {'condition_index', 'well_index', 'timepoint_index', 'total_PDO_projected_area_um2'}
    missing = required - set(out.columns)
    if missing:
        raise ValueError('Missing required longitudinal columns: ' + ', '.join(sorted(missing)))

    if 'elapsed_time' not in out.columns:
        out['elapsed_time'] = pd.to_numeric(out['timepoint_index'], errors='coerce') - 1.0
    out['elapsed_time'] = pd.to_numeric(out['elapsed_time'], errors='coerce')
    out['total_PDO_projected_area_um2'] = pd.to_numeric(
        out['total_PDO_projected_area_um2'], errors='coerce'
    ).fillna(0.0)

    keys = ['condition_index', 'well_index']
    first = (out.sort_values(['condition_index', 'well_index', 'timepoint_index'])
             .groupby(keys, as_index=False)
             .first()[keys + ['elapsed_time', 'total_PDO_projected_area_um2']]
             .rename(columns={
                 'elapsed_time': 'baseline_elapsed_time',
                 'total_PDO_projected_area_um2': 'baseline_total_PDO_area_um2_calc',
             }))
    out = out.merge(first, on=keys, how='left', validate='many_to_one')

    base = out['baseline_total_PDO_area_um2_calc'].astype(float)
    current = out['total_PDO_projected_area_um2'].astype(float)
    valid = base > 0
    out['relative_total_PDO_area_vs_baseline'] = np.where(valid, current / base, np.nan)
    out['percent_area_change_vs_baseline'] = np.where(
        valid, 100.0 * (out['relative_total_PDO_area_vs_baseline'] - 1.0), np.nan
    )
    dt = out['elapsed_time'] - out['baseline_elapsed_time']
    out['log_area_growth_rate_per_time'] = np.where(
        valid & (current > 0) & (dt > 0), np.log(current / base) / dt, np.nan
    )
    return out


def final_response_table(ldf: pd.DataFrame) -> pd.DataFrame:
    if ldf is None or not len(ldf):
        return pd.DataFrame()
    x = add_response_metrics(ldf)
    group_cols = ['condition_index', 'well_index']
    if 'condition' in x.columns:
        group_cols.insert(1, 'condition')
    final = (x.sort_values('timepoint_index')
             .groupby(group_cols, as_index=False, sort=False)
             .tail(1)
             .reset_index(drop=True))
    return final


def _four_pl(x, bottom, top, midpoint, hill):
    x = np.asarray(x, dtype=float)
    midpoint = max(float(midpoint), np.finfo(float).eps)
    return bottom + (top - bottom) / (1.0 + np.power(np.maximum(x, 0.0) / midpoint, hill))


def analyse_growth_and_dose(ldf: pd.DataFrame, outdir: Path, dose_unit: str, time_unit: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = add_response_metrics(ldf)
    valid = x[x['baseline_total_PDO_area_um2_calc'] > 0].copy()
    if valid.empty:
        return {'status': 'skipped', 'reason': 'No wells had a positive baseline PDO area.'}

    grouping = ['condition_index', 'condition', 'timepoint_index', 'timepoint', 'elapsed_time']
    rows = []
    for keys, g in valid.groupby(grouping, dropna=False, sort=False):
        vals = pd.to_numeric(g['relative_total_PDO_area_vs_baseline'], errors='coerce').dropna()
        if vals.empty:
            continue
        rows.append({
            'condition_index': keys[0], 'condition': keys[1],
            'timepoint_index': keys[2], 'timepoint': keys[3], 'elapsed_time': keys[4],
            'n_tracked_wells': int(len(vals)),
            'median_relative_area': float(vals.median()),
            'q25_relative_area': float(vals.quantile(0.25)),
            'q75_relative_area': float(vals.quantile(0.75)),
            'mean_relative_area': float(vals.mean()),
        })
    growth = pd.DataFrame(rows)
    growth.to_csv(outdir/'growth_rate_population_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for cond, g in growth.groupby('condition', sort=False):
        g = g.sort_values('elapsed_time')
        xx = pd.to_numeric(g['elapsed_time'], errors='coerce').to_numpy(float)
        med = g['median_relative_area'].to_numpy(float)
        lo = g['q25_relative_area'].to_numpy(float)
        hi = g['q75_relative_area'].to_numpy(float)
        ax.plot(xx, med, marker='o', label=str(cond))
        ax.fill_between(xx, lo, hi, alpha=0.15)
    ax.axhline(1.0, ls='--', linewidth=1)
    ax.set_xlabel(f'Elapsed time ({time_unit})')
    ax.set_ylabel('Relative total PDO area vs baseline')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir/'growth_trajectory_by_condition.png', dpi=300)
    plt.close(fig)

    final = final_response_table(x)
    if 'concentration' not in final.columns:
        return {'status': 'partial', 'reason': 'Growth trajectories created; no numeric concentration metadata was supplied.'}

    dose = final.copy()
    dose['concentration'] = pd.to_numeric(dose['concentration'], errors='coerce')
    dose = dose[dose['concentration'].notna() & dose['relative_total_PDO_area_vs_baseline'].notna()]
    if dose.empty:
        return {'status': 'partial', 'reason': 'Growth trajectories created; no usable numeric concentrations were supplied.'}

    summary = (dose.groupby('concentration', as_index=False)
               .agg(n_tracked_wells=('relative_total_PDO_area_vs_baseline', 'size'),
                    median_relative_area=('relative_total_PDO_area_vs_baseline', 'median'),
                    q25_relative_area=('relative_total_PDO_area_vs_baseline', lambda s: s.quantile(0.25)),
                    q75_relative_area=('relative_total_PDO_area_vs_baseline', lambda s: s.quantile(0.75))))
    summary.to_csv(outdir/'dose_response_summary.csv', index=False)

    xx = summary['concentration'].to_numpy(float)
    yy = summary['median_relative_area'].to_numpy(float)
    yerr = np.vstack([
        yy - summary['q25_relative_area'].to_numpy(float),
        summary['q75_relative_area'].to_numpy(float) - yy,
    ])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(xx, yy, yerr=yerr, fmt='o', capsize=3, label='Median ± IQR')

    unique_x = np.unique(xx[np.isfinite(xx)])
    positive = unique_x[unique_x > 0]
    fit_row = {'dose_unit': dose_unit}
    if len(unique_x) >= 4 and len(positive) >= 3 and np.nanmax(yy) > np.nanmin(yy):
        try:
            p0 = [float(np.nanmin(yy)), float(np.nanmax(yy)), float(np.median(positive)), 1.0]
            popt, _ = curve_fit(
                _four_pl, xx, yy, p0=p0,
                bounds=([-np.inf, -np.inf, np.finfo(float).eps, 0.05],
                        [np.inf, np.inf, np.inf, 10.0]),
                maxfev=20000,
            )
            grid = np.linspace(0.0 if np.any(xx == 0) else float(np.min(positive)), float(np.max(unique_x)), 400)
            ax.plot(grid, _four_pl(grid, *popt), label='4-parameter logistic fit')
            fit_row.update({
                'bottom': float(popt[0]), 'top': float(popt[1]),
                'response_midpoint_concentration': float(popt[2]),
                'hill_slope': float(popt[3]),
                'note': 'Imaging-response 4PL midpoint; not automatically a GI50 or viability IC50.',
            })
        except Exception as exc:
            fit_row['fit_error'] = str(exc)
    else:
        fit_row['fit_error'] = 'At least four distinct concentrations, including at least three >0, are required for a 4PL fit.'
    pd.DataFrame([fit_row]).to_csv(outdir/'dose_response_4PL_fit.csv', index=False)

    if len(positive):
        ax.set_xscale('symlog', linthresh=max(float(np.min(positive))/10.0, np.finfo(float).eps))
    ax.set_xlabel(f'Concentration ({dose_unit})')
    ax.set_ylabel('Final relative total PDO area vs baseline')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir/'dose_response_final_relative_area.png', dpi=300)
    plt.close(fig)
    return {'status': 'complete'}


def analyse_waterfall(ldf: pd.DataFrame, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    final = final_response_table(ldf)
    final = final[final['percent_area_change_vs_baseline'].notna()].copy()
    if final.empty:
        return {'status': 'skipped', 'reason': 'No wells had a positive baseline PDO area.'}
    final.to_csv(outdir/'waterfall_final_well_responses.csv', index=False)
    for cond, g in final.groupby('condition', sort=False):
        g = g.sort_values('percent_area_change_vs_baseline').reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.bar(np.arange(len(g)), g['percent_area_change_vs_baseline'].to_numpy(float), width=1.0)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel('Tracked microwells, sorted by response')
        ax.set_ylabel('Final PDO area change from baseline (%)')
        ax.set_title(str(cond))
        fig.tight_layout()
        fig.savefig(outdir/f'waterfall_{_safe_name(cond)}.png', dpi=300)
        plt.close(fig)
    return {'status': 'complete'}


def analyse_heatmaps(ldf: pd.DataFrame, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = add_response_metrics(ldf)
    x = x[x['relative_total_PDO_area_vs_baseline'].notna()].copy()
    if x.empty:
        return {'status': 'skipped', 'reason': 'No wells had a positive baseline PDO area.'}
    for cond, g in x.groupby('condition', sort=False):
        pivot = g.pivot_table(index='well_index', columns='timepoint_index', values='relative_total_PDO_area_vs_baseline', aggfunc='first')
        if pivot.empty:
            continue
        final_col = sorted(pivot.columns)[-1]
        pivot = pivot.assign(_sort=pivot[final_col]).sort_values('_sort').drop(columns='_sort')
        labels = (g[['timepoint_index', 'timepoint']].drop_duplicates()
                  .set_index('timepoint_index')['timepoint'].to_dict())
        fig, ax = plt.subplots(figsize=(7.0, min(12.0, max(4.5, 0.025*len(pivot)+2.5))))
        im = ax.imshow(pivot.to_numpy(float), aspect='auto', interpolation='nearest')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([labels.get(c, str(c)) for c in pivot.columns])
        ax.set_xlabel('Time point')
        ax.set_ylabel('Tracked microwells (sorted by final response)')
        ax.set_title(str(cond))
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Relative total PDO area vs baseline')
        fig.tight_layout()
        fig.savefig(outdir/f'heatmap_{_safe_name(cond)}.png', dpi=300)
        plt.close(fig)
        pivot.to_csv(outdir/f'heatmap_matrix_{_safe_name(cond)}.csv')
    return {'status': 'complete'}


def _ecdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    return v, np.arange(1, len(v)+1) / len(v)


def analyse_distributions(ldf: pd.DataFrame, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = add_response_metrics(ldf)
    x = x[x['percent_area_change_vs_baseline'].notna()].copy()
    if x.empty:
        return {'status': 'skipped', 'reason': 'No wells had a positive baseline PDO area.'}
    x.to_csv(outdir/'all_longitudinal_response_distributions.csv', index=False)

    for tp_idx, tg in x.groupby('timepoint_index', sort=True):
        label_text = str(tg['timepoint'].iloc[0])
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        drawn = False
        for cond, g in tg.groupby('condition', sort=False):
            vals = pd.to_numeric(g['percent_area_change_vs_baseline'], errors='coerce').dropna().to_numpy(float)
            if len(vals):
                vx, vy = _ecdf(vals)
                ax.step(vx, vy, where='post', label=str(cond))
                drawn = True
        if drawn:
            ax.axvline(0, linewidth=1)
            ax.set_xlabel('PDO area change from baseline (%)')
            ax.set_ylabel('Empirical cumulative fraction')
            ax.set_title(label_text)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(outdir/f'ECDF_time_{int(tp_idx):02d}_{_safe_name(label_text)}.png', dpi=300)
        plt.close(fig)

    final = final_response_table(x)
    groups, labels = [], []
    for cond, g in final.groupby('condition', sort=False):
        vals = g['percent_area_change_vs_baseline'].dropna().to_numpy(float)
        if len(vals):
            groups.append(vals)
            labels.append(str(cond))
    if groups:
        fig, ax = plt.subplots(figsize=(max(7.0, 1.1*len(groups)+3), 4.8))
        ax.violinplot(groups, showmedians=True, showextrema=True)
        ax.set_xticks(np.arange(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.axhline(0, linewidth=1)
        ax.set_ylabel('Final PDO area change from baseline (%)')
        fig.tight_layout()
        fig.savefig(outdir/'final_response_violin_by_condition.png', dpi=300)
        plt.close(fig)
    return {'status': 'complete'}


def analyse_responder_classes(ldf: pd.DataFrame, outdir: Path, responder_max_pct: float, resistant_min_pct: float):
    if responder_max_pct >= resistant_min_pct:
        raise ValueError('Responder threshold must be lower than the resistant threshold.')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    final = final_response_table(ldf)
    final = final[final['percent_area_change_vs_baseline'].notna()].copy()
    if final.empty:
        return {'status': 'skipped', 'reason': 'No wells had a positive baseline PDO area.'}

    final['response_class'] = np.select(
        [final['percent_area_change_vs_baseline'] <= responder_max_pct,
         final['percent_area_change_vs_baseline'] >= resistant_min_pct],
        ['Responder', 'Resistant'],
        default='Intermediate',
    )
    final['responder_max_pct_threshold'] = float(responder_max_pct)
    final['resistant_min_pct_threshold'] = float(resistant_min_pct)
    final.to_csv(outdir/'well_response_classification.csv', index=False)

    summary = final.groupby(['condition', 'response_class']).size().rename('well_count').reset_index()
    totals = final.groupby('condition').size().rename('total_classified_wells').reset_index()
    summary = summary.merge(totals, on='condition', how='left')
    summary['percentage'] = 100.0 * summary['well_count'] / summary['total_classified_wells']
    summary.to_csv(outdir/'response_class_summary.csv', index=False)

    pivot = summary.pivot(index='condition', columns='response_class', values='percentage').fillna(0.0)
    order = [c for c in ['Responder', 'Intermediate', 'Resistant'] if c in pivot.columns]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1*len(pivot)+3), 4.8))
    bottom = np.zeros(len(pivot))
    for cls in order:
        vals = pivot[cls].to_numpy(float)
        ax.bar(np.arange(len(pivot)), vals, bottom=bottom, label=cls)
        bottom += vals
    ax.set_xticks(np.arange(len(pivot)))
    ax.set_xticklabels(pivot.index.astype(str), rotation=30, ha='right')
    ax.set_ylabel('Classified tracked wells (%)')
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir/'response_classification_by_condition.png', dpi=300)
    plt.close(fig)
    return {'status': 'complete'}


def _bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    ps = p[finite]
    order = np.argsort(ps)
    ranked = ps[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    restored = np.empty(m)
    restored[order] = np.clip(adj, 0, 1)
    out[idx] = restored
    return out


def analyse_psc_association(ldf: pd.DataFrame, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x = add_response_metrics(ldf)
    if 'RFP_PSC_stromal_cells_present' in x.columns:
        x = x[x['RFP_PSC_stromal_cells_present'].fillna(False).astype(bool)]
    if 'PSC_like_focus_count' not in x.columns:
        return {'status': 'skipped', 'reason': 'No PSC-like focus count column is present.'}
    x['PSC_like_focus_count'] = pd.to_numeric(x['PSC_like_focus_count'], errors='coerce')
    x = x[x['PSC_like_focus_count'].notna() & x['percent_area_change_vs_baseline'].notna()].copy()
    if x.empty:
        return {'status': 'skipped', 'reason': 'No tracked wells had both PSC counts and a valid PDO response.'}

    rows = []
    for keys, g in x.groupby(['condition', 'timepoint_index', 'timepoint'], sort=False):
        pair = g[['PSC_like_focus_count', 'percent_area_change_vs_baseline']].dropna()
        if len(pair) >= 3 and pair['PSC_like_focus_count'].nunique() >= 2 and pair['percent_area_change_vs_baseline'].nunique() >= 2:
            rho, pval = spearmanr(pair['PSC_like_focus_count'], pair['percent_area_change_vs_baseline'])
        else:
            rho, pval = np.nan, np.nan
        rows.append({
            'condition': keys[0], 'timepoint_index': keys[1], 'timepoint': keys[2],
            'n_wells': int(len(pair)),
            'spearman_rho': float(rho) if np.isfinite(rho) else np.nan,
            'p_value_unadjusted': float(pval) if np.isfinite(pval) else np.nan,
        })
    stats = pd.DataFrame(rows)
    stats['p_value_BH_FDR'] = _bh_adjust(stats['p_value_unadjusted'].to_numpy(float))
    stats['interpretation_note'] = 'Exploratory well-level association; wells are not independent biological replicates.'
    stats.to_csv(outdir/'PSC_response_spearman_by_condition_time.csv', index=False)

    final = final_response_table(x)
    final.to_csv(outdir/'PSC_final_well_response_data.csv', index=False)
    for cond, g in final.groupby('condition', sort=False):
        if len(g) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.scatter(g['PSC_like_focus_count'], g['percent_area_change_vs_baseline'], alpha=0.65)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel('PSC-like red fluorescent foci in well')
        ax.set_ylabel('Final PDO area change from baseline (%)')
        ax.set_title(str(cond))
        fig.tight_layout()
        fig.savefig(outdir/f'PSC_vs_response_{_safe_name(cond)}.png', dpi=300)
        plt.close(fig)
    return {'status': 'complete'}


def run_selected_analyses(ldf: pd.DataFrame, outdir: Path, selected, *, dose_unit='nM', time_unit='days',
                          responder_max_pct=None, resistant_min_pct=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected = list(selected or [])
    enriched = add_response_metrics(ldf)
    enriched.to_csv(outdir/'longitudinal_response_metrics.csv', index=False)
    results = []

    for name in selected:
        module_dir = outdir / _safe_name(name)
        try:
            if name == ANALYSIS_GROWTH:
                result = analyse_growth_and_dose(enriched, module_dir, dose_unit, time_unit)
            elif name == ANALYSIS_WATERFALL:
                result = analyse_waterfall(enriched, module_dir)
            elif name == ANALYSIS_HEATMAP:
                result = analyse_heatmaps(enriched, module_dir)
            elif name == ANALYSIS_DISTRIBUTIONS:
                result = analyse_distributions(enriched, module_dir)
            elif name == ANALYSIS_CLASSIFICATION:
                if responder_max_pct is None or resistant_min_pct is None:
                    result = {'status': 'skipped', 'reason': 'Responder/resistant thresholds were not supplied.'}
                else:
                    result = analyse_responder_classes(enriched, module_dir, float(responder_max_pct), float(resistant_min_pct))
            elif name == ANALYSIS_PSC:
                result = analyse_psc_association(enriched, module_dir)
            else:
                result = {'status': 'skipped', 'reason': 'Unknown analysis module.'}
        except Exception as exc:
            result = {'status': 'error', 'reason': str(exc)}
        results.append({'analysis': name, **result})

    manifest = pd.DataFrame(results)
    manifest.to_csv(outdir/'analysis_manifest.csv', index=False)
    return manifest
