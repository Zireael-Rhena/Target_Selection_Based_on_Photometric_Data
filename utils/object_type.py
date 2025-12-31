"""
检查 CLAUDS 数据集中的 COMPACT 字段
"""

import numpy as np
from astropy.table import Table
from collections import Counter


def check_COMPACT(path: str, field_name: str = "Field"):
    """
    检查 COMPACT 字段的值分布
    """
    
    print("=" * 70)
    print(f"COMPACT Analysis:  {field_name}")
    print(f"File: {path}")
    print("=" * 70)
    
    # 加载数据
    tab = Table.read(path, hdu=1)
    n_total = len(tab)
    print(f"\nTotal sources: {n_total: ,}")
    
    # 检查 COMPACT 列
    if 'COMPACT' not in tab.colnames:
        print("COMPACT column not found!")
        return None
    
    COMPACT = np.array(tab['COMPACT'])
    
    # 基本信息
    print(f"\nColumn info:")
    print(f"  dtype: {COMPACT.dtype}")
    print(f"  shape: {COMPACT.shape}")
    
    # 统计唯一值
    unique_vals, counts = np.unique(COMPACT, return_counts=True)
    
    # 按数量排序
    sorted_idx = np.argsort(-counts)
    unique_vals = unique_vals[sorted_idx]
    counts = counts[sorted_idx]
    
    print(f"\nUnique values: {len(unique_vals)}")
    print(f"\nValue distribution:")
    print("-" * 50)
    print(f"{'Value':<20} | {'Count': >12} | {'Percentage':>10}")
    print("-" * 50)
    
    for val, count in zip(unique_vals, counts):
        pct = count / n_total * 100
        # 处理不同类型的显示
        if isinstance(val, (int, np.integer)):
            val_str = str(val)
        elif isinstance(val, (float, np.floating)):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
        print(f"{val_str:<20} | {count: >12,} | {pct: >9.2f}%")
    
    # 如果是整数类型，尝试解释含义
    print("\n" + "-" * 50)
    print("Possible interpretation:")
    print("-" * 50)
    
    # 常见的 COMPACT 编码
    common_meanings = {
        0: "Galaxy (normal)",
        1: "Star",
        2: "QSO/AGN",
        3: "Unknown/Other",
        -1: "Failed fit / No classification",
        -99: "No data / Invalid",
    }
    
    for val in unique_vals[: 10]:  # 只显示前10个
        if val in common_meanings:
            print(f"  {val}: {common_meanings[val]}")
        else:
            print(f"  {val}: Unknown meaning")
    
    # 与其他标志的交叉分析
    print("\n" + "=" * 70)
    print("Cross-analysis with other flags")
    print("=" * 70)
    
    # COMPACT vs COMPACT
    if 'COMPACT' in tab.colnames:
        print("\n  COMPACT vs COMPACT:")
        compact = np.array(tab['COMPACT'])
        for val in unique_vals[: 5]:
            mask = (COMPACT == val)
            n_compact = np.sum(compact[mask] == 1)
            n_total_val = np.sum(mask)
            if n_total_val > 0:
                pct = n_compact / n_total_val * 100
                print(f"    COMPACT={val}: {n_compact:,}/{n_total_val:,} compact ({pct:.1f}%)")
    
    # COMPACT vs CLASS_STAR
    if 'CLASS_STAR_HSC_I' in tab. colnames:
        print("\n  COMPACT vs CLASS_STAR_HSC_I (mean value):")
        class_star = np.array(tab['CLASS_STAR_HSC_I'])
        # 过滤有效值
        valid_cs = (class_star >= 0) & (class_star <= 1)
        
        for val in unique_vals[:5]: 
            mask = (COMPACT == val) & valid_cs
            if np.sum(mask) > 0:
                mean_cs = np.mean(class_star[mask])
                std_cs = np.std(class_star[mask])
                print(f"    COMPACT={val}: CLASS_STAR mean={mean_cs:.3f} ± {std_cs:.3f}")
    
    # COMPACT vs Z_BEST (photo-z)
    if 'Z_BEST' in tab. colnames:
        print("\n  COMPACT vs Z_BEST (photo-z distribution):")
        z_best = np.array(tab['Z_BEST'])
        valid_z = (z_best >= 0) & (z_best < 10)
        
        for val in unique_vals[:5]:
            mask = (COMPACT == val) & valid_z
            if np.sum(mask) > 100: 
                mean_z = np.mean(z_best[mask])
                median_z = np.median(z_best[mask])
                high_z = np.sum(z_best[mask] > 2) / np.sum(mask) * 100
                print(f"    COMPACT={val}: z_mean={mean_z:.2f}, z_median={median_z:.2f}, z>2: {high_z:.1f}%")
    
    # COMPACT vs STAR_FORMING
    if 'STAR_FORMING' in tab.colnames:
        print("\n  COMPACT vs STAR_FORMING:")
        sf = np.array(tab['STAR_FORMING'])
        for val in unique_vals[:5]: 
            mask = (COMPACT == val)
            n_sf = np.sum(sf[mask] == 1)
            n_total_val = np.sum(mask)
            if n_total_val > 0:
                pct = n_sf / n_total_val * 100
                print(f"    COMPACT={val}: {n_sf:,}/{n_total_val:,} star-forming ({pct:.1f}%)")
    
    return COMPACT, unique_vals, counts


# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    
    # COSMOS
    print("\n" + "#" * 80)
    print("# COSMOS")
    print("#" * 80 + "\n")
    
    obj_cosmos, vals_cosmos, counts_cosmos = check_COMPACT(
        "../../data_Clauds/COSMOS_6bands-SExtractor-Lephare.fits",
        "COSMOS"
    )
    
    print("\n\n")
    
    # XMM
    print("#" * 80)
    print("# XMM-LSS")
    print("#" * 80 + "\n")
    
    obj_xmm, vals_xmm, counts_xmm = check_COMPACT(
        "../../data_Clauds/XMMLSS_6bands-SExtractor-Lephare.fits",
        "XMM-LSS"
    )
    
    # 两天区对比
    print("\n\n")
    print("#" * 80)
    print("# COMPARISON")
    print("#" * 80)
    
    print("\n  COMPACT distribution comparison:")
    print(f"  {'Value':<10} | {'COSMOS':>15} | {'XMM-LSS':>15}")
    print("  " + "-" * 45)
    
    all_vals = set(vals_cosmos) | set(vals_xmm)
    cosmos_dict = dict(zip(vals_cosmos, counts_cosmos))
    xmm_dict = dict(zip(vals_xmm, counts_xmm))
    
    for val in sorted(all_vals):
        c_count = cosmos_dict.get(val, 0)
        x_count = xmm_dict. get(val, 0)
        c_pct = c_count / len(obj_cosmos) * 100 if c_count > 0 else 0
        x_pct = x_count / len(obj_xmm) * 100 if x_count > 0 else 0
        print(f"  {val:<10} | {c_count:>10,} ({c_pct: >4.1f}%) | {x_count:>10,} ({x_pct:>4.1f}%)")