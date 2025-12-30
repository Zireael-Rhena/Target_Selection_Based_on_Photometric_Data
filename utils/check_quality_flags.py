"""
检查 CLAUDS 数据集中的质量标志逻辑
分析各个 flag 之间的关系
"""

import numpy as np
from astropy.table import Table
from collections import Counter


def check_quality_flags(path:  str, field_name: str = "Field"):
    """
    检查 CLAUDS 数据集中的质量标志
    """
    
    print("=" * 70)
    print(f"Quality Flags Analysis:  {field_name}")
    print(f"File: {path}")
    print("=" * 70)
    
    # 加载数据
    print("\nLoading data...")
    tab = Table.read(path, hdu=1, memmap=True)
    n_total = len(tab)
    print(f"Total sources: {n_total: ,}")
    
    # ==========================================
    # 1. 单波段质量标志分析
    # ==========================================
    print("\n" + "=" * 70)
    print("1. PER-BAND QUALITY FLAGS")
    print("=" * 70)
    
    # 选择几个代表性波段
    bands = ['HSC-G', 'HSC-I', 'MegaCam-u', 'MegaCam-uS']
    flag_types = ['hasBadPhotometry', 'isDuplicated', 'isNoData', 
                  'isSky', 'isParent', 'notObserved', 'isClean']
    
    for band in bands:
        # 检查该波段是否存在
        test_col = f'isClean_{band}'
        if test_col not in tab.colnames:
            print(f"\n  [{band}] - Not available in this catalog")
            continue
            
        print(f"\n  [{band}]")
        print("  " + "-" * 50)
        
        # 统计各标志
        flag_counts = {}
        for flag_type in flag_types:
            col_name = f'{flag_type}_{band}'
            if col_name in tab.colnames:
                values = np.array(tab[col_name])
                n_true = np.sum(values == True) if values.dtype == bool else np.sum(values == 1)
                pct = n_true / n_total * 100
                flag_counts[flag_type] = n_true
                print(f"    {flag_type: <20}:  {n_true:>10,} ({pct: >6.2f}%)")
        
        # 检查 isClean 是否等于其他标志的组合
        if all(f'{ft}_{band}' in tab. colnames for ft in flag_types):
            print(f"\n    Checking if isClean = NOT(hasBadPhotometry | isDuplicated | isNoData | isSky | isParent | notObserved)...")
            
            is_clean = np.array(tab[f'isClean_{band}']).astype(bool)
            
            # 计算预期的 clean 条件
            bad_photo = np.array(tab[f'hasBadPhotometry_{band}']).astype(bool)
            duplicated = np.array(tab[f'isDuplicated_{band}']).astype(bool)
            no_data = np.array(tab[f'isNoData_{band}']).astype(bool)
            is_sky = np.array(tab[f'isSky_{band}']).astype(bool)
            is_parent = np.array(tab[f'isParent_{band}']).astype(bool)
            not_observed = np.array(tab[f'notObserved_{band}']).astype(bool)
            
            # 假设 isClean = NOT(任何问题)
            expected_clean = ~(bad_photo | duplicated | no_data | is_sky | is_parent | not_observed)
            
            # 比较
            match = np.sum(is_clean == expected_clean)
            mismatch = n_total - match
            
            print(f"    Match with expected: {match:,} ({match/n_total*100:.2f}%)")
            print(f"    Mismatch:  {mismatch:,}")
            
            if mismatch > 0 and mismatch < 1000:
                # 分析不匹配的情况
                mismatch_mask = is_clean != expected_clean
                print(f"\n    Mismatch analysis (first 10):")
                mismatch_idx = np.where(mismatch_mask)[0][:10]
                for idx in mismatch_idx: 
                    print(f"      idx={idx}:  isClean={is_clean[idx]}, "
                          f"bad={bad_photo[idx]}, dup={duplicated[idx]}, "
                          f"nodata={no_data[idx]}, sky={is_sky[idx]}, "
                          f"parent={is_parent[idx]}, notobs={not_observed[idx]}")
    
    # ==========================================
    # 2. 全局质量标志分析
    # ==========================================
    print("\n" + "=" * 70)
    print("2. GLOBAL QUALITY FLAGS")
    print("=" * 70)
    
    global_flags = ['isCompact', 'isOutsideMask', 'isStarTemp', 'isStar', 'FLAG_FIELD_BINARY']
    
    for flag in global_flags:
        if flag not in tab.colnames:
            print(f"\n  {flag}:  Not found")
            continue
            
        values = np.array(tab[flag])
        
        print(f"\n  {flag}:")
        print(f"    dtype: {values.dtype}")
        print(f"    shape: {values.shape}")
        
        if values.dtype == bool or (values.dtype in [np.int64, np.int32, np.uint8] and len(values. shape) == 1):
            # 简单布尔或整数标志
            n_true = np.sum(values == True) if values.dtype == bool else np.sum(values == 1)
            n_false = np.sum(values == False) if values.dtype == bool else np.sum(values == 0)
            print(f"    True:   {n_true:>10,} ({n_true/n_total*100:.2f}%)")
            print(f"    False: {n_false:>10,} ({n_false/n_total*100:.2f}%)")
            
        elif len(values.shape) > 1:
            # 多维数组 (如 FLAG_FIELD_BINARY)
            print(f"    This is a multi-dimensional flag")
            for i in range(values.shape[1]):
                col = values[:, i]
                n_true = np.sum(col. astype(bool))
                print(f"    Column [{i}]: True={n_true:>10,} ({n_true/n_total*100:.2f}%)")
    
    # ==========================================
    # 3. 分析 isStar, isStarTemp, isCompact 的关系
    # ==========================================
    print("\n" + "=" * 70)
    print("3. STAR/COMPACT CLASSIFICATION RELATIONSHIP")
    print("=" * 70)
    
    if all(f in tab.colnames for f in ['isStar', 'isStarTemp', 'isCompact']):
        is_star = np.array(tab['isStar']).astype(bool)
        is_star_temp = np.array(tab['isStarTemp']).astype(bool)
        is_compact = np.array(tab['isCompact']).astype(bool)
        
        # 交叉统计
        print("\n  Cross-tabulation:")
        print("  " + "-" * 50)
        
        # isStar vs isStarTemp
        print(f"\n  isStar vs isStarTemp:")
        print(f"    Both True:       {np.sum(is_star & is_star_temp):>10,}")
        print(f"    Both False:     {np.sum(~is_star & ~is_star_temp):>10,}")
        print(f"    Star only:      {np.sum(is_star & ~is_star_temp):>10,}")
        print(f"    StarTemp only:  {np.sum(~is_star & is_star_temp):>10,}")
        
        # isStar vs isCompact
        print(f"\n  isStar vs isCompact:")
        print(f"    Both True:       {np.sum(is_star & is_compact):>10,}")
        print(f"    Both False:     {np. sum(~is_star & ~is_compact):>10,}")
        print(f"    Star only:      {np.sum(is_star & ~is_compact):>10,}")
        print(f"    Compact only:    {np.sum(~is_star & is_compact):>10,}")
        
        # isStarTemp vs isCompact
        print(f"\n  isStarTemp vs isCompact:")
        print(f"    Both True:      {np.sum(is_star_temp & is_compact):>10,}")
        print(f"    Both False:     {np.sum(~is_star_temp & ~is_compact):>10,}")
        print(f"    StarTemp only:  {np.sum(is_star_temp & ~is_compact):>10,}")
        print(f"    Compact only:   {np.sum(~is_star_temp & is_compact):>10,}")
        
        # 检查是否 isStar = isStarTemp AND isCompact
        expected_star = is_star_temp & is_compact
        match_star = np.sum(is_star == expected_star)
        print(f"\n  Check: isStar = isStarTemp & isCompact ? ")
        print(f"    Match: {match_star:,} ({match_star/n_total*100:.2f}%)")
        
        # 检查是否 isStar 是 isStarTemp 的子集
        is_subset = np.all(is_star <= is_star_temp)
        print(f"\n  Check: isStar ⊆ isStarTemp ?  {is_subset}")
        
        # 检查是否 isStar 是 isCompact 的子集
        is_subset2 = np.all(is_star <= is_compact)
        print(f"  Check: isStar ⊆ isCompact ? {is_subset2}")
    
    # ==========================================
    # 4. FLAG_FIELD_BINARY 详细分析
    # ==========================================
    print("\n" + "=" * 70)
    print("4. FLAG_FIELD_BINARY ANALYSIS")
    print("=" * 70)
    
    if 'FLAG_FIELD_BINARY' in tab.colnames:
        flag_field = np.array(tab['FLAG_FIELD_BINARY'])
        print(f"\n  Shape: {flag_field.shape}")
        print(f"  dtype: {flag_field.dtype}")
        
        # 根据 README:  
        # COSMOS: FLAG_FIELD_BINARY[:,0] & FLAG_FIELD_BINARY[:,1]
        # XMM: FLAG_FIELD_BINARY[:,0] & FLAG_FIELD_BINARY[:,2]
        
        n_cols = flag_field.shape[1] if len(flag_field.shape) > 1 else 1
        print(f"\n  Column statistics:")
        for i in range(n_cols):
            col = flag_field[:, i] if n_cols > 1 else flag_field
            n_true = np.sum(col. astype(bool))
            print(f"    Column [{i}]: True = {n_true:>10,} ({n_true/n_total*100:.2f}%)")
        
        if n_cols >= 3:
            col0 = flag_field[:, 0]. astype(bool)
            col1 = flag_field[:, 1].astype(bool)
            col2 = flag_field[:, 2].astype(bool)
            
            # COSMOS 选择:  col0 & col1
            cosmos_sel = col0 & col1
            # XMM 选择: col0 & col2
            xmm_sel = col0 & col2
            
            print(f"\n  Field selections (based on README):")
            print(f"    COSMOS (col0 & col1): {np.sum(cosmos_sel):>10,} ({np.sum(cosmos_sel)/n_total*100:.2f}%)")
            print(f"    XMM (col0 & col2):    {np.sum(xmm_sel):>10,} ({np.sum(xmm_sel)/n_total*100:.2f}%)")
    
    # ==========================================
    # 5. isOutsideMask 与 FLAG_FIELD_BINARY 的关系
    # ==========================================
    print("\n" + "=" * 70)
    print("5. isOutsideMask vs FLAG_FIELD_BINARY")
    print("=" * 70)
    
    if 'isOutsideMask' in tab.colnames and 'FLAG_FIELD_BINARY' in tab.colnames:
        outside_mask = np.array(tab['isOutsideMask']).astype(bool)
        flag_field = np.array(tab['FLAG_FIELD_BINARY'])
        
        print(f"\n  isOutsideMask:")
        print(f"    True:  {np.sum(outside_mask):>10,}")
        print(f"    False: {np.sum(~outside_mask):>10,}")
        
        # 检查关系
        if len(flag_field. shape) > 1:
            col0 = flag_field[:, 0].astype(bool)
            
            # isOutsideMask 可能与 FLAG_FIELD_BINARY[:,0] 相关
            print(f"\n  Relationship with FLAG_FIELD_BINARY[:,0]:")
            print(f"    Both True:       {np.sum(outside_mask & col0):>10,}")
            print(f"    Both False:      {np. sum(~outside_mask & ~col0):>10,}")
            print(f"    OutsideMask only:{np.sum(outside_mask & ~col0):>10,}")
            print(f"    Flag0 only:      {np.sum(~outside_mask & col0):>10,}")
            
            # 检查是否互补
            is_complement = np.sum(outside_mask == ~col0) / n_total * 100
            print(f"\n  isOutsideMask ≈ NOT(FLAG_FIELD_BINARY[:,0]) ? {is_complement:.2f}% match")
    
    # ==========================================
    # 6. 推荐的清洁样本选择
    # ==========================================
    print("\n" + "=" * 70)
    print("6. RECOMMENDED CLEAN SAMPLE SELECTION")
    print("=" * 70)
    
    # 基于 README 的建议
    if 'FLAG_FIELD_BINARY' in tab.colnames:
        flag_field = np.array(tab['FLAG_FIELD_BINARY'])
        col0 = flag_field[:, 0].astype(bool)
        col1 = flag_field[:, 1].astype(bool) if flag_field.shape[1] > 1 else np.ones(n_total, dtype=bool)
        col2 = flag_field[:, 2].astype(bool) if flag_field.shape[1] > 2 else np. ones(n_total, dtype=bool)
        
        # 检查是否有 MASK 列
        if 'MASK' in tab. colnames:
            mask = np.array(tab['MASK'])
            mask_ok = (mask == 0)
        else:
            mask_ok = np.ones(n_total, dtype=bool)
        
        # 非恒星
        if 'isStar' in tab.colnames:
            not_star = ~np.array(tab['isStar']).astype(bool)
        else:
            not_star = np.ones(n_total, dtype=bool)
        
        # 组合条件
        clean_cosmos = mask_ok & col0 & col1 & not_star
        clean_xmm = mask_ok & col0 & col2 & not_star
        
        print(f"\n  Clean sample (COSMOS style): mask_ok & col0 & col1 & not_star")
        print(f"    Count: {np.sum(clean_cosmos):>10,} ({np.sum(clean_cosmos)/n_total*100:.2f}%)")
        
        print(f"\n  Clean sample (XMM style): mask_ok & col0 & col2 & not_star")
        print(f"    Count: {np.sum(clean_xmm):>10,} ({np.sum(clean_xmm)/n_total*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    
    return tab


# ==========================================
# 运行分析
# ==========================================
if __name__ == "__main__": 
    
    # 分析 COSMOS
    print("\n" + "#" * 80)
    print("# COSMOS FIELD")
    print("#" * 80)
    tab_cosmos = check_quality_flags(
        "../../data_Clauds/COSMOS-HSCpipe-Phosphoros.fits",
        "COSMOS"
    )
    
    print("\n\n")
    
    # 分析 XMM
    print("#" * 80)
    print("# XMM-LSS FIELD")
    print("#" * 80)
    tab_xmm = check_quality_flags(
        "../../data_Clauds/XMMLSS-HSCpipe-Phosphoros.fits", 
        "XMM-LSS"
    )