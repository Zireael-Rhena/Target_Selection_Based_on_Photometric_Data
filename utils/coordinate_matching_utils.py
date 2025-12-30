"""
坐标匹配与 Flux-Mag 转换验证工具
用于 CLAUDS 和 DESI 数据的交叉匹配分析
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from typing import Dict, List, Tuple, Optional


def auto_detect_coord_columns(table: Table) -> Tuple[str, str]:
    """
    自动检测表格中的坐标列名
    
    Parameters
    ----------
    table : astropy.table.Table
        输入表格
        
    Returns
    -------
    ra_col, dec_col : tuple of str
        检测到的 RA 和 DEC 列名
    """
    # RA 候选列名 (按优先级排序)
    ra_candidates = ['RA', 'TARGET_RA', 'ra', 'RAJ2000', 'RA_J2000', 
                     'ALPHA_J2000', 'MEAN_FIBER_RA', 'PLATE_RA']
    # DEC 候选列名
    dec_candidates = ['DEC', 'TARGET_DEC', 'dec', 'Dec', 'DEJ2000', 'DE_J2000',
                      'DELTA_J2000', 'MEAN_FIBER_DEC', 'PLATE_DEC']
    
    ra_col = None
    dec_col = None
    
    for candidate in ra_candidates:
        if candidate in table.colnames:
            ra_col = candidate
            break
    
    for candidate in dec_candidates: 
        if candidate in table.colnames:
            dec_col = candidate
            break
    
    return ra_col, dec_col


def coordinate_match_and_verify(
    path_selected: str,
    path_raw: str,
    field_name: str = "Field",
    match_radius: float = 1.0,
    mag_col_candidates: List[str] = None,
    flux_col_candidates: List[str] = None,
    ra_col_selected: str = None,  # 改为 None，支持自动检测
    dec_col_selected: str = None,
    ra_col_raw: str = None,
    dec_col_raw: str = None,
    save_plots: bool = False,
    show_plots: bool = True
) -> Dict:
    """
    对两个天文数据集进行坐标匹配，并验证 Flux -> Mag 转换公式。
    
    Parameters
    ----------
    path_selected : str
        筛选后的数据文件路径 (包含 Magnitude)
    path_raw : str
        原始 CLAUDS 测光数据路径 (包含 Flux)
    field_name : str
        天区名称，用于图表标题和保存文件名
    match_radius : float
        匹配半径，单位 arcsec，默认 1.0
    mag_col_candidates : list
        Magnitude 列名候选列表
    flux_col_candidates : list
        Flux 列名候选列表
    ra_col_selected, dec_col_selected : str or None
        筛选数据中的坐标列名，None 表示自动检测
    ra_col_raw, dec_col_raw : str or None
        原始数据中的坐标列名，None 表示自动检测
    save_plots : bool
        是否保存图片
    show_plots : bool
        是否显示图片
        
    Returns
    -------
    results : dict
        包含匹配结果和验证结果的字典
    """
    
    # 默认列名候选
    if mag_col_candidates is None: 
        mag_col_candidates = ['uS', 'MAG_APER_2s_uS', 'u', 'MAG_APER_2s_u']
    if flux_col_candidates is None:
        flux_col_candidates = [
            'FLUX_APER_2_MegaCam-uS', 
            'FLUX_APER_2_MegaCam-u',
            'FLUX_APER_2_HSC-G'
        ]
    
    results = {
        'field_name': field_name,
        'n_selected': 0,
        'n_raw': 0,
        'n_matched': 0,
        'match_rate': 0.0,
        'separation_stats': {},
        'zeropoint': None,
        'zeropoint_std': None,
        'mag_col': None,
        'flux_col': None,
        'coord_cols_selected': None,
        'coord_cols_raw': None
    }
    
    # ==========================================
    # 1. 加载数据
    # ==========================================
    print("=" * 60)
    print(f"[{field_name}] Step 1: Loading Data Files...")
    print("=" * 60)
    
    try:
        tab_selected = Table.read(path_selected, hdu=1)
        print(f"✓ Selected catalog:  {len(tab_selected)} rows")
        print(f"  Path: {path_selected}")
        results['n_selected'] = len(tab_selected)
    except Exception as e:
        print(f"✗ Error loading selected catalog: {e}")
        return results
    
    try: 
        tab_raw = Table. read(path_raw, hdu=1, memmap=True)
        print(f"✓ Raw CLAUDS catalog: {len(tab_raw)} rows")
        print(f"  Path: {path_raw}")
        results['n_raw'] = len(tab_raw)
    except Exception as e:
        print(f"✗ Error loading raw catalog: {e}")
        return results
    
    # ==========================================
    # 2. 检测/确认坐标列名
    # ==========================================
    print(f"\n[{field_name}] Step 2: Detecting Coordinate Columns...")
    print("-" * 40)
    
    # 自动检测或使用指定的列名
    if ra_col_selected is None or dec_col_selected is None: 
        detected_ra, detected_dec = auto_detect_coord_columns(tab_selected)
        ra_col_selected = ra_col_selected or detected_ra
        dec_col_selected = dec_col_selected or detected_dec
        print(f"  [Selected] Auto-detected:  RA='{ra_col_selected}', DEC='{dec_col_selected}'")
    else:
        print(f"  [Selected] Using specified:  RA='{ra_col_selected}', DEC='{dec_col_selected}'")
    
    if ra_col_raw is None or dec_col_raw is None: 
        detected_ra, detected_dec = auto_detect_coord_columns(tab_raw)
        ra_col_raw = ra_col_raw or detected_ra
        dec_col_raw = dec_col_raw or detected_dec
        print(f"  [Raw] Auto-detected: RA='{ra_col_raw}', DEC='{dec_col_raw}'")
    else:
        print(f"  [Raw] Using specified: RA='{ra_col_raw}', DEC='{dec_col_raw}'")
    
    # 检查列名是否存在
    if ra_col_selected is None or ra_col_selected not in tab_selected.colnames:
        print(f"  ✗ Error: RA column '{ra_col_selected}' not found in selected catalog")
        print(f"    Available columns: {tab_selected. colnames[: 15]}...")
        return results
    if dec_col_selected is None or dec_col_selected not in tab_selected.colnames:
        print(f"  ✗ Error: DEC column '{dec_col_selected}' not found in selected catalog")
        return results
    if ra_col_raw is None or ra_col_raw not in tab_raw.colnames:
        print(f"  ✗ Error: RA column '{ra_col_raw}' not found in raw catalog")
        print(f"    Available columns: {tab_raw.colnames[:15]}...")
        return results
    if dec_col_raw is None or dec_col_raw not in tab_raw.colnames:
        print(f"  ✗ Error: DEC column '{dec_col_raw}' not found in raw catalog")
        return results
    
    results['coord_cols_selected'] = (ra_col_selected, dec_col_selected)
    results['coord_cols_raw'] = (ra_col_raw, dec_col_raw)
    
    # ==========================================
    # 3. 预览坐标数据
    # ==========================================
    print(f"\n[{field_name}] Step 3: Coordinate Preview")
    print("-" * 40)
    
    print(f"\n【Selected】前3行 ({ra_col_selected}, {dec_col_selected}):")
    for i in range(min(3, len(tab_selected))):
        ra = tab_selected[ra_col_selected][i]
        dec = tab_selected[dec_col_selected][i]
        print(f"  Row {i}: RA={ra:.6f}, DEC={dec:.6f}")
    
    print(f"\n【Raw】前3行 ({ra_col_raw}, {dec_col_raw}):")
    for i in range(min(3, len(tab_raw))):
        ra = tab_raw[ra_col_raw][i]
        dec = tab_raw[dec_col_raw][i]
        print(f"  Row {i}: RA={ra:.6f}, DEC={dec:.6f}")
    
    # ==========================================
    # 4. 坐标匹配
    # ==========================================
    print(f"\n[{field_name}] Step 4: Coordinate Matching...")
    print("-" * 40)
    
    coords_selected = SkyCoord(
        ra=tab_selected[ra_col_selected] * u.deg,
        dec=tab_selected[dec_col_selected] * u.deg
    )
    coords_raw = SkyCoord(
        ra=tab_raw[ra_col_raw] * u.deg,
        dec=tab_raw[dec_col_raw] * u.deg
    )
    
    print(f"  Selected coordinates:  {len(coords_selected)}")
    print(f"  Raw coordinates: {len(coords_raw)}")
    print(f"\n  Matching...")
    
    idx, d2d, _ = match_coordinates_sky(coords_selected, coords_raw)
    
    print(f"  ✓ Matching complete!")
    
    # ==========================================
    # 5. 分析匹配结果
    # ==========================================
    print(f"\n[{field_name}] Step 5: Matching Results Analysis")
    print("-" * 40)
    
    separations = d2d.to(u.arcsec).value
    
    results['separation_stats'] = {
        'min': np.min(separations),
        'max': np.max(separations),
        'mean': np.mean(separations),
        'median': np.median(separations)
    }
    
    print(f"\n  角距离统计 (arcsec):")
    print(f"    Min:    {results['separation_stats']['min']:.4f}")
    print(f"    Max:    {results['separation_stats']['max']:.4f}")
    print(f"    Mean:   {results['separation_stats']['mean']:.4f}")
    print(f"    Median: {results['separation_stats']['median']:.4f}")
    
    # 不同阈值下的匹配数量
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"\n  不同匹配半径下的成功数量:")
    for thresh in thresholds:
        n_matched = np.sum(separations < thresh)
        pct = n_matched / len(separations) * 100
        print(f"    < {thresh: 4.1f} arcsec: {n_matched:6d} ({pct:.2f}%)")
    
    # 使用指定阈值
    good_match_mask = separations < match_radius
    n_good_match = np.sum(good_match_mask)
    results['n_matched'] = n_good_match
    results['match_rate'] = n_good_match / len(tab_selected) * 100
    
    print(f"\n  ★ 使用阈值 {match_radius} arcsec:")
    print(f"    匹配成功: {n_good_match} / {len(tab_selected)} ({results['match_rate']:.2f}%)")
    
    # ==========================================
    # 6. 验证 Flux -> Mag 转换
    # ==========================================
    print(f"\n[{field_name}] Step 6: Flux to Mag Conversion Verification")
    print("-" * 40)
    
    matched_raw_idx = idx[good_match_mask]
    
    # 查找可用列名
    mag_col = None
    for col in mag_col_candidates:
        if col in tab_selected.colnames:
            mag_col = col
            break
    
    flux_col = None
    for col in flux_col_candidates: 
        if col in tab_raw.colnames:
            flux_col = col
            break
    
    print(f"  Mag 列:  {mag_col}")
    print(f"  Flux 列: {flux_col}")
    
    results['mag_col'] = mag_col
    results['flux_col'] = flux_col
    
    # 初始化变量
    zp_mean = None
    zp_std = None
    mag_valid = None
    flux_valid = None
    
    if mag_col and flux_col:
        # 提取数据
        mag_values = np.array(tab_selected[mag_col][good_match_mask])
        flux_values = np.array(tab_raw[flux_col][matched_raw_idx])
        
        # 过滤有效数据
        valid_mask = (
            np.isfinite(mag_values) &
            np.isfinite(flux_values) &
            (flux_values > 0) &
            (mag_values > 15) &
            (mag_values < 35)
        )
        
        mag_valid = mag_values[valid_mask]
        flux_valid = flux_values[valid_mask]
        
        print(f"\n  有效数据点: {len(mag_valid)}")
        
        if len(mag_valid) > 100:
            # 测试不同 ZeroPoint
            print(f"\n  测试不同 ZeroPoint:")
            print(f"  {'ZP': >6} | {'Mean Residual':>14} | {'Std': >10}")
            print("  " + "-" * 40)
            
            for zp in [22.5, 23.9, 27.0, 31.4]:
                mag_calc = zp - 2.5 * np.log10(flux_valid)
                residual = mag_valid - mag_calc
                print(f"  {zp:6.1f} | {np.mean(residual):+14.4f} | {np.std(residual):10.4f}")
            
            # 反推 ZeroPoint
            inferred_zp = mag_valid + 2.5 * np. log10(flux_valid)
            zp_mean = np.mean(inferred_zp)
            zp_std = np.std(inferred_zp)
            
            results['zeropoint'] = zp_mean
            results['zeropoint_std'] = zp_std
            
            print(f"\n  ★ 推断的 ZeroPoint: {zp_mean:.4f} ± {zp_std:.4f}")
        else:
            print(f"\n  ⚠ 有效数据点不足 (<100)，无法可靠推断 ZeroPoint")
    else:
        print(f"\n  ⚠ 无法找到 Mag/Flux 列进行转换验证 (跳过此步骤)")
        if mag_col is None: 
            print(f"    Selected catalog columns: {tab_selected.colnames[:15]}...")
        if flux_col is None:
            print(f"    Raw catalog columns:  {tab_raw.colnames[: 15]}...")
    
    # ==========================================
    # 7. 绘图 (仅当有有效数据时)
    # ==========================================
    if (save_plots or show_plots) and mag_valid is not None and len(mag_valid) > 100:
        _plot_verification(
            separations, match_radius,
            mag_valid, flux_valid, zp_mean,
            mag_col, field_name,
            save_plots, show_plots
        )
    
    # ==========================================
    # 8. 总结
    # ==========================================
    print(f"\n[{field_name}] Summary")
    print("=" * 60)
    
    zp_str = f"{zp_mean:.2f}" if zp_mean is not None else "N/A"
    zp_std_str = f"{zp_std:.4f}" if zp_std is not None else "N/A"
    
    print(f"""
  1. 坐标列: 
     - Selected: ({ra_col_selected}, {dec_col_selected})
     - Raw: ({ra_col_raw}, {dec_col_raw})
     
  2. 坐标匹配结果: 
     - 总源数: {len(tab_selected)}
     - < {match_radius} arcsec 匹配:  {n_good_match} ({results['match_rate']:.1f}%)
     
  3.  Flux -> Mag 转换公式:
     m = {zp_str} - 2.5 * log10(Flux)
     ZP Std:  {zp_std_str}
""")
    
    return results


def _plot_verification(
    separations:  np.ndarray,
    match_radius: float,
    mag_valid: np.ndarray,
    flux_valid: np. ndarray,
    zp_mean: float,
    mag_col: str,
    field_name: str,
    save_plots: bool,
    show_plots: bool
):
    """绘制验证图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{field_name}:  Coordinate Matching & Flux-Mag Verification', fontsize=14)
    
    # 图1: 角距离分布 (全范围)
    ax1 = axes[0, 0]
    ax1.hist(separations, bins=100, range=(0, 5), edgecolor='black', alpha=0.7)
    ax1.axvline(match_radius, color='r', linestyle='--', label=f'{match_radius} arcsec threshold')
    ax1.set_xlabel('Angular Separation (arcsec)')
    ax1.set_ylabel('Count')
    ax1.set_title('Matching Distance Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 角距离分布 (放大)
    ax2 = axes[0, 1]
    ax2.hist(separations[separations < match_radius], bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Angular Separation (arcsec)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Zoomed:  Separation < {match_radius} arcsec')
    ax2.grid(True, alpha=0.3)
    
    # 图3: Mag 对比
    mag_calc_best = zp_mean - 2.5 * np.log10(flux_valid)
    
    ax3 = axes[1, 0]
    ax3.scatter(mag_valid, mag_calc_best, alpha=0.3, s=5)
    lims = [min(mag_valid. min(), mag_calc_best.min()),
            max(mag_valid.max(), mag_calc_best.max())]
    ax3.plot(lims, lims, 'r--', lw=2, label='1:1 line')
    ax3.set_xlabel(f'Mag from Catalog ({mag_col})')
    ax3.set_ylabel(f'Mag from Flux (ZP={zp_mean:.2f})')
    ax3.set_title('Magnitude Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 残差分布
    residual_best = mag_valid - mag_calc_best
    
    ax4 = axes[1, 1]
    ax4.hist(residual_best, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Residual (Catalog - Calculated)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Residual:  Mean={np.mean(residual_best):.4f}, Std={np.std(residual_best):.4f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'{field_name}_verification.png'
        plt.savefig(filename, dpi=150)
        print(f"\n  ✓ 图片已保存: {filename}")
    
    if show_plots: 
        plt.show()
    else:
        plt.close()


def batch_verify(configs: List[Dict], **common_kwargs) -> Dict[str, Dict]:
    """
    批量验证多个天区的数据
    
    Parameters
    ----------
    configs : list of dict
        每个字典包含 'path_selected', 'path_raw', 'field_name' 等参数
        可选参数:  'ra_col_selected', 'dec_col_selected', 'ra_col_raw', 'dec_col_raw'
    **common_kwargs : 
        所有天区共用的参数
        
    Returns
    -------
    all_results : dict
        以 field_name 为 key 的结果字典
    """
    all_results = {}
    
    for config in configs:
        merged_kwargs = {**common_kwargs, **config}
        field_name = merged_kwargs.get('field_name', 'Unknown')
        
        print("\n" + "=" * 70)
        print(f"Processing: {field_name}")
        print("=" * 70 + "\n")
        
        results = coordinate_match_and_verify(**merged_kwargs)
        all_results[field_name] = results
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"\n{'Field':<15} | {'Selected':<10} | {'Matched':<10} | {'Rate': >8} | {'ZeroPoint':>12}")
    print("-" * 65)
    
    for field, res in all_results.items():
        zp_str = f"{res['zeropoint']:.2f}" if res['zeropoint'] is not None else "N/A"
        print(f"{field: <15} | {res['n_selected']:<10} | {res['n_matched']:<10} | {res['match_rate']: >7.2f}% | {zp_str: >12}")
    
    return all_results
