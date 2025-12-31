"""
预览新的 CLAUDS SExtractor-Lephare 数据集
分析可用列并确定需要提取的物理量
"""

import numpy as np
from astropy.table import Table
from astropy.io import fits


def preview_catalog(path:  str, field_name: str = "Field", n_rows: int = 5):
    """
    预览 CLAUDS catalog 的结构和前几行数据
    """
    
    print("=" * 80)
    print(f"Catalog Preview:  {field_name}")
    print(f"File: {path}")
    print("=" * 80)
    
    # 加载数据
    tab = Table.read(path, hdu=1)
    n_total = len(tab)
    print(f"\nTotal sources: {n_total: ,}")
    print(f"Total columns: {len(tab.colnames)}")
    
    # ==========================================
    # 1. 列分类
    # ==========================================
    print("\n" + "=" * 80)
    print("1.  COLUMN CLASSIFICATION")
    print("=" * 80)
    
    # 按类别分组列名
    categories = {
        'Coordinates': [],
        'Morphology': [],
        'Magnitudes (Total)': [],
        'Magnitudes (Aperture 2s)': [],
        'Magnitudes (Aperture 3s)': [],
        'Magnitude Errors': [],
        'Photo-z': [],
        'Quality Flags': [],
        'Other': []
    }
    
    for col in tab.colnames:
        col_lower = col.lower()
        if col in ['ID', 'RA', 'DEC', 'TRACT', 'PATCH']: 
            categories['Coordinates'].append(col)
        elif col in ['A_WORLD', 'B_WORLD', 'KRON_RADIUS', 'THETA_WORLD', 
                     'ELONGATION', 'ELLIPTICITY', 'FWHM_WORLD_HSC_I',
                     'MU_MAX_HSC_I', 'CLASS_STAR_HSC_I', 'FLUX_RADIUS_0.25_HSC_I',
                     'FLUX_RADIUS_0.5_HSC_I', 'FLUX_RADIUS_0.75_HSC_I']: 
            categories['Morphology']. append(col)
        elif col in ['FUV', 'NUV', 'u', 'uS', 'g', 'r', 'i', 'z', 'y']:
            categories['Magnitudes (Total)'].append(col)
        elif 'MAG_APER_2s' in col and 'ERR' not in col:
            categories['Magnitudes (Aperture 2s)'].append(col)
        elif 'MAG_APER_3s' in col and 'ERR' not in col:
            categories['Magnitudes (Aperture 3s)'].append(col)
        elif '_err' in col_lower or 'ERR' in col: 
            categories['Magnitude Errors'].append(col)
        elif col in ['Z_BEST', 'Z_BEST68_LOW', 'Z_BEST68_HIGH', 'CHI_BEST',
                     'CHI_STAR', 'CHI_QSO', 'MOD_BEST', 'MOD_STAR', 'MOD_QSO',
                     'Z_ML', 'Z_ML68_LOW', 'Z_ML68_HIGH', 'Z_SEC', 'Z_QSO', 'ZPHOT']:
            categories['Photo-z'].append(col)
        elif col in ['MASK', 'FLAG_FIELD', 'FLAG_FIELD_BINARY', 'CLEAN', 
                     'OBJ_TYPE', 'COMPACT', 'STAR_FORMING', 'ST_TRAIL', 'CONTEXT']:
            categories['Quality Flags'].append(col)
        else:
            categories['Other'].append(col)
    
    for cat_name, cols in categories.items():
        if cols:
            print(f"\n  {cat_name}:")
            print(f"    {cols}")
    
    # ==========================================
    # 2. 数据类型和基本统计
    # ==========================================
    print("\n" + "=" * 80)
    print("2. KEY COLUMNS - DATA TYPES & BASIC STATS")
    print("=" * 80)
    
    key_cols = ['u', 'uS', 'g', 'r', 'i', 'z', 'y',
                'MAG_APER_2s_uS', 'MAG_APER_2s_g', 'MAG_APER_2s_i',
                'Z_BEST', 'ZPHOT', 'CLASS_STAR_HSC_I',
                'MASK', 'FLAG_FIELD', 'CLEAN', 'COMPACT']
    
    print(f"\n  {'Column':<25} | {'dtype':<10} | {'min': >10} | {'max':>10} | {'N_valid':>12} | {'N_invalid':>12}")
    print("  " + "-" * 95)
    
    for col in key_cols:
        if col not in tab.colnames:
            print(f"  {col: <25} | {'N/A':<10}")
            continue
        
        data = np.array(tab[col])
        dtype_str = str(data.dtype)[:10]
        
        # 处理不同类型
        if data.dtype == bool:
            n_true = np.sum(data)
            print(f"  {col:<25} | {dtype_str:<10} | {'True:':<10} {n_true:>10,} | {'False:':<10} {n_total - n_true:>10,}")
        elif len(data. shape) > 1:
            print(f"  {col:<25} | {dtype_str:<10} | {'shape:':<10} {str(data.shape):>20}")
        else:
            # 数值型
            valid_mask = np.isfinite(data) & (data > -90) & (data < 90)
            n_valid = np. sum(valid_mask)
            n_invalid = n_total - n_valid
            
            if n_valid > 0:
                valid_data = data[valid_mask]
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                print(f"  {col:<25} | {dtype_str:<10} | {min_val:>10.2f} | {max_val:>10.2f} | {n_valid:>12,} | {n_invalid:>12,}")
            else:
                print(f"  {col:<25} | {dtype_str:<10} | {'N/A': >10} | {'N/A':>10} | {n_valid:>12,} | {n_invalid:>12,}")
    
    # ==========================================
    # 3. 前 N 行数据预览
    # ==========================================
    print("\n" + "=" * 80)
    print(f"3. FIRST {n_rows} ROWS - SELECTED COLUMNS")
    print("=" * 80)
    
    # 选择要显示的列
    preview_cols_groups = [
        # 组1: 坐标和ID
        ['ID', 'RA', 'DEC'],
        # 组2: 总星等 (关键波段)
        ['u', 'uS', 'g', 'r', 'i', 'z'],
        # 组3: 孔径星等
        ['MAG_APER_2s_uS', 'MAG_APER_2s_g', 'MAG_APER_2s_r', 'MAG_APER_2s_i'],
        # 组4: 星等误差
        ['uS_err', 'g_err', 'r_err', 'i_err'],
        # 组5: Photo-z
        ['Z_BEST', 'ZPHOT', 'CHI_BEST'],
        # 组6: 质量标志
        ['MASK', 'FLAG_FIELD', 'CLEAN', 'COMPACT', 'CLASS_STAR_HSC_I']
    ]
    
    for group_idx, cols in enumerate(preview_cols_groups):
        # 过滤存在的列
        existing_cols = [c for c in cols if c in tab.colnames]
        if not existing_cols:
            continue
        
        print(f"\n  Group {group_idx + 1}:  {existing_cols}")
        print("  " + "-" * 80)
        
        for i in range(min(n_rows, n_total)):
            row_str = f"  Row {i}:  "
            values = []
            for col in existing_cols:
                val = tab[col][i]
                if isinstance(val, (float, np.floating)):
                    if np.isfinite(val) and abs(val) < 100:
                        values.append(f"{col}={val:.4f}")
                    else: 
                        values.append(f"{col}={val:.2e}")
                elif isinstance(val, (int, np.integer)):
                    values. append(f"{col}={val}")
                elif isinstance(val, (bool, np.bool_)):
                    values.append(f"{col}={val}")
                elif hasattr(val, '__len__') and len(val) > 1:
                    values.append(f"{col}={list(val)[:3]}...")
                else:
                    values.append(f"{col}={val}")
            print(row_str + ", ".join(values))
    
    # ==========================================
    # 4. 质量标志分析
    # ==========================================
    print("\n" + "=" * 80)
    print("4. QUALITY FLAGS ANALYSIS")
    print("=" * 80)
    
    # MASK
    if 'MASK' in tab. colnames:
        mask_vals = np.array(tab['MASK'])
        print(f"\n  MASK:")
        unique_vals, counts = np.unique(mask_vals, return_counts=True)
        for val, count in zip(unique_vals[: 10], counts[:10]):
            pct = count / n_total * 100
            print(f"    Value {val}: {count:>10,} ({pct:.2f}%)")
    
    # FLAG_FIELD
    if 'FLAG_FIELD' in tab.colnames:
        ff_vals = np.array(tab['FLAG_FIELD'])
        print(f"\n  FLAG_FIELD:")
        unique_vals, counts = np.unique(ff_vals, return_counts=True)
        for val, count in zip(unique_vals[:10], counts[:10]):
            pct = count / n_total * 100
            print(f"    Value {val}: {count:>10,} ({pct:.2f}%)")
    
    # FLAG_FIELD_BINARY
    if 'FLAG_FIELD_BINARY' in tab.colnames:
        ffb = np.array(tab['FLAG_FIELD_BINARY'])
        print(f"\n  FLAG_FIELD_BINARY:")
        print(f"    Shape: {ffb.shape}")
        if len(ffb.shape) > 1:
            for i in range(ffb.shape[1]):
                n_true = np.sum(ffb[: , i]. astype(bool))
                print(f"    Column [{i}]: True = {n_true:>10,} ({n_true/n_total*100:.2f}%)")
    
    # CLEAN
    if 'CLEAN' in tab.colnames:
        clean_vals = np.array(tab['CLEAN'])
        print(f"\n  CLEAN:")
        if clean_vals.dtype == bool:
            n_clean = np.sum(clean_vals)
            print(f"    True:   {n_clean:>10,} ({n_clean/n_total*100:.2f}%)")
            print(f"    False: {n_total - n_clean:>10,} ({(n_total-n_clean)/n_total*100:.2f}%)")
        else:
            unique_vals, counts = np.unique(clean_vals, return_counts=True)
            for val, count in zip(unique_vals[: 10], counts[:10]):
                pct = count / n_total * 100
                print(f"    Value {val}: {count:>10,} ({pct:.2f}%)")
    
    # COMPACT
    if 'COMPACT' in tab.colnames:
        compact_vals = np.array(tab['COMPACT'])
        print(f"\n  COMPACT (likely star/point-source indicator):")
        unique_vals, counts = np.unique(compact_vals, return_counts=True)
        for val, count in zip(unique_vals[: 10], counts[:10]):
            pct = count / n_total * 100
            print(f"    Value {val}: {count:>10,} ({pct:.2f}%)")
    
    # CLASS_STAR_HSC_I
    if 'CLASS_STAR_HSC_I' in tab.colnames:
        cs_vals = np.array(tab['CLASS_STAR_HSC_I'])
        valid_cs = cs_vals[np.isfinite(cs_vals)]
        print(f"\n  CLASS_STAR_HSC_I (0=galaxy, 1=star):")
        print(f"    Min: {np.min(valid_cs):.4f}")
        print(f"    Max: {np.max(valid_cs):.4f}")
        print(f"    Mean: {np.mean(valid_cs):.4f}")
        print(f"    Likely stars (>0.8): {np.sum(valid_cs > 0.8):,} ({np.sum(valid_cs > 0.8)/len(valid_cs)*100:.2f}%)")
        print(f"    Likely galaxies (<0.2): {np.sum(valid_cs < 0.2):,} ({np.sum(valid_cs < 0.2)/len(valid_cs)*100:.2f}%)")
    
    # ==========================================
    # 5. Non-detection 分析 (星等 = 99 或 -99)
    # ==========================================
    print("\n" + "=" * 80)
    print("5. NON-DETECTION ANALYSIS")
    print("=" * 80)
    
    mag_cols = ['u', 'uS', 'g', 'r', 'i', 'z', 'y']
    print(f"\n  {'Band':<10} | {'N_detected': >12} | {'N_nondetect':>12} | {'Det_rate':>10}")
    print("  " + "-" * 55)
    
    for col in mag_cols:
        if col not in tab.colnames:
            continue
        data = np.array(tab[col])
        # 通常 non-detection 用 99, -99, 或 NaN 表示
        detected = np.isfinite(data) & (data > 0) & (data < 50)
        n_det = np.sum(detected)
        n_nondet = n_total - n_det
        det_rate = n_det / n_total * 100
        print(f"  {col:<10} | {n_det:>12,} | {n_nondet: >12,} | {det_rate:>9.2f}%")
    
    print("\n" + "=" * 80)
    print("Preview Complete")
    print("=" * 80)
    
    return tab
