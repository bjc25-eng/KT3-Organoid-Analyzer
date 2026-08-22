import ast
import math
from pathlib import Path

import numpy as np
import pytest

import aws_export_final_containment_boundary_review as review
import aws_pdo_full_component_qc as component_qc


CONDITIONS = list(component_qc.CONDITIONS)


class FakePlanes:
    def __init__(self, gfp):
        self.data = np.zeros((3, *gfp.shape), np.uint8)
        self.data[0] = gfp
        self.shape_cyx = self.data.shape

    def read(self, channel, ys, xs):
        return self.data[channel, ys, xs]


def authoritative_row(fraction, index, *, condition=None, component_id=None,
                      many=False, well_id=None):
    condition = condition or CONDITIONS[index % len(CONDITIONS)]
    well_id = str(well_id if well_id is not None else 1000 + index)
    row = {field: '' for field in component_qc.PDO_FIELDS}
    row.update({
        'condition_id': condition,
        'condition_name': component_qc.CONDITIONS[condition]['condition_name'],
        'dose': component_qc.CONDITIONS[condition]['dose'],
        'well_id': well_id, 'pdo_number_in_well': 1,
        'containment_qc_status': component_qc.QC_STATUS,
        'full_component_fraction_inside_final_well': fraction,
        'full_component_fraction_outside_final_well': 1-fraction,
        'full_component_fraction_inside_production_0p86r': max(0, fraction-.05),
        'full_component_area_px2': 100+index,
        'production_PDO_projected_area_px2': 40+index,
        'production_PDO_projected_area_um2': 40+index,
        'production_PDO_equivalent_circular_diameter_um': 5,
        'production_PDO_centroid_x_px_fullres': 50,
        'production_PDO_centroid_y_px_fullres': 50,
        'pixel_size_x_um': 1, 'pixel_size_y_um': 1,
        'unmasked_component_id': component_id or f'gfpcomp_{index:064x}',
        'many_production_PDOs_to_one_unmasked_component': many,
        'well_x_px_fullres': 50, 'well_y_px_fullres': 50, 'well_radius_px': 10,
    })
    return row


def full_sampling_fixture():
    rows=[]; index=0
    ranges=((.30,.399),(.401,.449),(.451,.499),(.501,.549),(.551,.649))
    for low,high in ranges:
        for position in range(12):
            fraction=low+(high-low)*position/11
            rows.append(authoritative_row(fraction,index)); index+=1
    for position in range(12):
        rows.append(authoritative_row(.70+position*.005,index,component_id=f'many_{position}',many=True)); index+=1
    rows.append(authoritative_row(.90,index,condition=component_qc.DMSO_CONDITION,well_id=606)); index+=1
    rows.append(authoritative_row(.91,index,condition=component_qc.DMSO_CONDITION,well_id=624))
    return rows


def well(well_id='1', x=50, y=50, radius=10):
    return {'well_id':well_id,'x_px_fullres':x,'y_px_fullres':y,'radius_px':radius,
            'PDO_count':1,'hex_array_member':True}


def production_pdo(component, well_id='1'):
    area=component.area_px2
    return {'well_id':well_id,'pdo_number_in_well':1,
            'centroid_x_px_fullres':component.centroid_x_fullres,
            'centroid_y_px_fullres':component.centroid_y_fullres,
            'projected_area_px2':area,'projected_area_um2':area,
            'equivalent_circular_diameter_um':2*math.sqrt(area/math.pi)}


def replay_fixture(long=False):
    gfp=np.zeros((201,201),np.uint8)
    if long:
        gfp[98:103,25:176]=255
    else:
        gfp[98:103,95:116]=255
    planes=FakePlanes(gfp); target=well(x=100,y=100,radius=10)
    clipped=component_qc.reproduce_production_components(planes,target,201,201,255,3)[0]
    pdo=production_pdo(clipped)
    if long:
        component_id='unused_for_incomplete'
    else:
        full,status=component_qc.recover_unmasked_component(planes,clipped,target,201,201,255,3)
        assert status['full_component_extent_status']=='complete'
        component_id=component_qc.component_hash(CONDITIONS[1],full)[0]
    row=authoritative_row(.5,1,condition=CONDITIONS[1],component_id=component_id,well_id=1)
    row.update(production_PDO_centroid_x_px_fullres=pdo['centroid_x_px_fullres'],
               production_PDO_centroid_y_px_fullres=pdo['centroid_y_px_fullres'],
               production_PDO_projected_area_px2=pdo['projected_area_px2'],
               production_PDO_projected_area_um2=pdo['projected_area_um2'],
               production_PDO_equivalent_circular_diameter_um=pdo['equivalent_circular_diameter_um'])
    return planes,target,pdo,row


def test_sampling_exact_fixture_counts_and_maximum_62():
    selected=review.select_review_rows(full_sampling_fixture())
    assert len(selected)==62
    counts={reason:sum(reason in row['sampling_reasons'].split(';') for row in selected)
            for reason in [name for name,_,_ in review.BIN_DEFINITIONS]
            + ['many_to_one_component_review','known_visual_failure_mandatory']}
    assert counts=={
        'closest_below_0p40':10,'bin_0p40_to_0p45':10,'bin_0p45_to_0p50':10,
        'bin_0p50_to_0p55':10,'bin_0p55_to_0p65':10,
        'many_to_one_component_review':10,'known_visual_failure_mandatory':2,
    }
    assert review.EXPECTED_MAXIMUM_DIAGNOSTICS==62


def test_closest_below_bin_selects_nearest_per_condition():
    rows=[authoritative_row(.39-i*.01,i,condition=CONDITIONS[i%2]) for i in range(8)]
    chosen=review._condition_balanced_closest_below(rows,4)
    assert sorted(row['full_component_fraction_inside_final_well'] for row in chosen)==[.36,.37,.38,.39]


def test_condition_balancing_differs_by_at_most_one_when_available():
    selected=review.select_review_rows(full_sampling_fixture())
    for name,_,_ in review.BIN_DEFINITIONS:
        rows=[row for row in selected if name in row['sampling_reasons'].split(';')]
        counts={condition:sum(row['condition_id']==condition for row in rows) for condition in CONDITIONS}
        nonzero=[value for value in counts.values() if value]
        assert max(nonzero)-min(nonzero)<=1


def test_many_to_one_uses_unique_components_nearest_half_and_balances():
    rows=[]
    for index in range(12):
        component=f'component_{index}'
        rows.append(authoritative_row(.41+index*.01,index,component_id=component,many=True))
        rows.append(authoritative_row(.8,index+100,condition=rows[-1]['condition_id'],component_id=component,many=True))
    selected=review._many_to_one_representatives(rows,10)
    assert len(selected)==10
    assert len({row['unmasked_component_id'] for row in selected})==10
    assert all(.40<=row['full_component_fraction_inside_final_well']<.60 for row in selected)


def test_known_failures_are_mandatory_and_not_exclusions():
    selected=review.select_review_rows(full_sampling_fixture())
    known={(row['well_id'],row['sampling_reasons']) for row in selected
           if 'known_visual_failure_mandatory' in row['sampling_reasons']}
    assert {item[0] for item in known}=={'606','624'}
    source=Path(review.__file__).read_text(encoding='utf-8')
    assert "exclusion_rule':None" in source
    assert 'exclusion_threshold' not in source


def test_missing_known_failure_fails_explicitly():
    rows=[row for row in full_sampling_fixture() if row['well_id']!='624']
    with pytest.raises(RuntimeError,match='well 624 is missing'):
        review.select_review_rows(rows)


def test_global_output_order_is_increasing_fraction_then_identity():
    selected=review.select_review_rows(full_sampling_fixture())
    assert selected==sorted(selected,key=review.review_sort_key)
    fractions=[float(row['full_component_fraction_inside_final_well']) for row in selected]
    assert fractions==sorted(fractions)


def test_exact_hash_match_passes_mask_replay():
    planes,target,pdo,row=replay_fixture()
    result,clipped,full,bbox,mask=review.replay_verified_masks(
        row,target,pdo,planes,201,201,255,3)
    assert result['mask_replay_status']=='verified'
    assert result['component_hash_verification_status']=='exact_hash_match'
    assert result['regenerated_unmasked_component_id']==row['unmasked_component_id']
    assert clipped is not None and full is not None and mask.size


def test_hash_mismatch_fails_diagnostic_verification():
    planes,target,pdo,row=replay_fixture(); row['unmasked_component_id']='gfpcomp_wrong'
    result,*_=review.replay_verified_masks(row,target,pdo,planes,201,201,255,3)
    assert result['mask_replay_status']=='failed'
    assert result['mask_replay_failure_reason']=='mask_replay_hash_mismatch'
    assert result['component_hash_verification_status']=='hash_mismatch'


def test_clipped_production_component_mismatch_fails_before_hash():
    planes,target,pdo,row=replay_fixture(); pdo['projected_area_px2']+=1
    result,*_=review.replay_verified_masks(row,target,pdo,planes,201,201,255,3)
    assert result['mask_replay_failure_reason']=='production_component_reproduction_failed'
    assert result['component_hash_verification_status']=='not_verified'


def test_crop_edge_incomplete_replay_is_not_verified():
    planes,target,pdo,row=replay_fixture(long=True)
    result,*_=review.replay_verified_masks(row,target,pdo,planes,201,201,255,3)
    assert result['mask_replay_failure_reason']=='crop_extent_incomplete'
    assert result['mask_replay_status']=='failed'
    assert result['component_hash_verification_status']=='not_verified'


def test_authoritative_values_drive_selection_and_are_not_recalculated():
    first=authoritative_row(.399,1,condition=CONDITIONS[0],well_id=606)
    second=authoritative_row(.2,2,condition=CONDITIONS[0],well_id=624)
    others=[authoritative_row(.1+i*.01,i+10) for i in range(15)]
    selected=review.select_review_rows(others+[first,second])
    assert first['full_component_fraction_inside_final_well']==.399
    assert any(row['well_id']=='606' and float(row['full_component_fraction_inside_final_well'])==.399
               for row in selected)
    tree=ast.parse(Path(review.__file__).read_text(encoding='utf-8'))
    called={node.func.id for node in ast.walk(tree)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    called.update(node.func.attr for node in ast.walk(tree)
                  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute))
    assert 'mask_containment' not in called
    assert 'quantify_condition_components' not in called


def test_diagnostic_layout_has_four_panels_and_existing_value_header():
    planes,target,pdo,row=replay_fixture()
    result,clipped,full,_,_=review.replay_verified_masks(row,target,pdo,planes,201,201,255,3)
    row.update(result,sampling_reasons='bin_0p50_to_0p55')
    bounds=(80,80,121,121); dic=np.zeros((41,41),np.uint8); gfp=planes.data[0,80:121,80:121]
    image=review.render_diagnostic(row,target,clipped,full,dic,gfp,bounds,(0,255),64)
    assert image.width==4*64+5*8
    assert image.height>64
    image.close()


def test_output_scope_and_prohibited_analysis_calls():
    source=Path(review.__file__).read_text(encoding='utf-8')
    assert review.OUTPUT_DIRECTORY=='final_containment_boundary_review'
    prohibited=('detect_psc(', 'HoughCircles(', 'bioformats2raw', 'watershed(',
                'quantify_condition_components(', 'mask_containment(')
    assert all(token not in source for token in prohibited)
    assert 'pdo_positive_crops_final_pdo_rfp' not in source
    assert review.REVIEW_STATUS=='visual_review_only_no_exclusion_rule'
