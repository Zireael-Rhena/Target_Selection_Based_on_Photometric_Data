"""
LBG (Lyman-Break Galaxy) 筛选工具
基于 DESI 光谱数据和 CLAUDS 测光数据进行高红移星系筛选
"""

import numpy as np
from astropy.table import Table
from typing import Dict, List, Tuple, Optional, Union
import warnings


def select_lbg_from_specz(
    table_or_path: Union[str, Table],
    z_threshold: float = 2.0,
    vi_quality_min: float = 2.5,
    rr_deltachi2_min: float = 9.0,
    vi_z_col: str = 'VI_Z',
    vi_quality_col: str = 'VI_QUALITY',
    rr_z_col: str = 'RR_Z',
    rr_deltachi2_col: str = 'RR_DELTACHI2',
    return_mask: bool = False,
    verbose: bool = True
) -> Union[Table, Tuple[Table, np.ndarray], Dict]: 
    """
    从 DESI 光谱数据中筛选高质量的 LBG 候选体。
    
    筛选逻辑 (基于 README 建议):
    - 条件 A: (VI_QUALITY >= 2.5) & (VI_Z > z_threshold)  [人工检查]
    - 条件 B: (RR_DELTACHI2 > 9) & (RR_Z > z_threshold)   [RedRock 自动拟合]
    - 最终:  条件 A OR 条件 B
    
    Parameters
    ----------
    table_or_path : str or astropy.table.Table
        输入数据，可以是 FITS 文件路径或 astropy Table
    z_threshold : float
        红移阈值，默认 2.0
    vi_quality_min : float
        VI_QUALITY 最小值，默认 2.5
    rr_deltachi2_min : float
        RR_DELTACHI2 最小值，默认 9.0
    vi_z_col, vi_quality_col :  str
        VI 相关列名
    rr_z_col, rr_deltachi2_col : str
        RedRock 相关列名
    return_mask : bool
        是否同时返回布尔掩码
    verbose : bool
        是否打印详细信息
        
    Returns
    -------
    如果 return_mask=False: 
        result : dict
            包含筛选结果的字典: 
            - 'lbg_table': 筛选后的 LBG Table
            - 'non_lbg_table': 非 LBG 但有可靠红移的 Table
            - 'stats': 统计信息字典
    如果 return_mask=True: 
        result, masks : tuple
            result 同上，masks 包含各种掩码
    """
    
    # ==========================================
    # 1. 加载数据
    # ==========================================
    if isinstance(table_or_path, str):
        if verbose:
            print(f"Loading:  {table_or_path}")
        table = Table. read(table_or_path, hdu=1)
    else:
        table = table_or_path. copy()
    
    n_total = len(table)
    if verbose:
        print(f"Total sources: {n_total}")
    
    # ==========================================
    # 2. 检查列名是否存在
    # ==========================================
    required_cols = [vi_z_col, vi_quality_col, rr_z_col, rr_deltachi2_col]
    missing_cols = [col for col in required_cols if col not in table.colnames]
    
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}\n"
                        f"Available columns: {table. colnames[: 20]}...")
    
    # ==========================================
    # 3. 提取数据并处理缺失值
    # ==========================================
    vi_z = np.array(table[vi_z_col], dtype=float)
    vi_quality = np.array(table[vi_quality_col], dtype=float)
    rr_z = np.array(table[rr_z_col], dtype=float)
    rr_deltachi2 = np.array(table[rr_deltachi2_col], dtype=float)
    
    # 将 NaN 替换为安全值 (不会满足条件)
    vi_z = np.nan_to_num(vi_z, nan=-99.0)
    vi_quality = np.nan_to_num(vi_quality, nan=-99.0)
    rr_z = np.nan_to_num(rr_z, nan=-99.0)
    rr_deltachi2 = np.nan_to_num(rr_deltachi2, nan=-99.0)
    
    # ==========================================
    # 4. 应用筛选条件
    # ==========================================
    # 条件 A: VI 高质量 + 高红移
    mask_vi_good = (vi_quality >= vi_quality_min)
    mask_vi_highz = (vi_z > z_threshold)
    mask_vi_lbg = mask_vi_good & mask_vi_highz
    
    # 条件 B: RR 高置信度 + 高红移
    mask_rr_good = (rr_deltachi2 > rr_deltachi2_min)
    mask_rr_highz = (rr_z > z_threshold)
    mask_rr_lbg = mask_rr_good & mask_rr_highz
    
    # 最终 LBG 掩码:  A OR B
    mask_lbg = mask_vi_lbg | mask_rr_lbg
    
    # 可靠红移但非 LBG (低红移 contaminants)
    mask_reliable = mask_vi_good | mask_rr_good
    mask_lowz = (vi_z <= z_threshold) & (vi_z > 0) | (rr_z <= z_threshold) & (rr_z > 0)
    mask_non_lbg = mask_reliable & mask_lowz & (~mask_lbg)
    
    # ==========================================
    # 5. 确定最佳红移值
    # ==========================================
    # 优先使用 VI_Z (人工检查)，其次使用 RR_Z
    z_best = np.where(mask_vi_good & (vi_z > 0), vi_z, rr_z)
    
    # 添加到表格
    table['Z_BEST'] = z_best
    table['IS_LBG'] = mask_lbg. astype(int)
    table['Z_SOURCE'] = np.where(mask_vi_good & (vi_z > 0), 'VI', 'RR')
    
    # ==========================================
    # 6. 统计信息
    # ==========================================
    stats = {
        'n_total': n_total,
        'n_vi_good': int(np.sum(mask_vi_good)),
        'n_rr_good': int(np. sum(mask_rr_good)),
        'n_reliable': int(np.sum(mask_reliable)),
        'n_vi_lbg': int(np.sum(mask_vi_lbg)),
        'n_rr_lbg': int(np.sum(mask_rr_lbg)),
        'n_lbg': int(np.sum(mask_lbg)),
        'n_non_lbg': int(np.sum(mask_non_lbg)),
        'lbg_fraction': np.sum(mask_lbg) / n_total * 100 if n_total > 0 else 0,
        'z_threshold': z_threshold,
        'vi_quality_min': vi_quality_min,
        'rr_deltachi2_min':  rr_deltachi2_min
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("LBG Selection Summary")
        print("=" * 50)
        print(f"\n  筛选条件:")
        print(f"    - z > {z_threshold}")
        print(f"    - VI_QUALITY >= {vi_quality_min} OR RR_DELTACHI2 > {rr_deltachi2_min}")
        print(f"\n  结果统计:")
        print(f"    - 总源数:               {stats['n_total']: 6d}")
        print(f"    - VI 质量合格:          {stats['n_vi_good']:6d}")
        print(f"    - RR 质量合格:          {stats['n_rr_good']:6d}")
        print(f"    - 可靠红移 (VI|RR):    {stats['n_reliable']:6d}")
        print(f"    - LBG (VI):            {stats['n_vi_lbg']:6d}")
        print(f"    - LBG (RR):            {stats['n_rr_lbg']:6d}")
        print(f"    - LBG 总数:            {stats['n_lbg']:6d} ({stats['lbg_fraction']:.1f}%)")
        print(f"    - 非 LBG (低红移):     {stats['n_non_lbg']:6d}")
    
    # ==========================================
    # 7. 构建返回结果
    # ==========================================
    lbg_table = table[mask_lbg]
    non_lbg_table = table[mask_non_lbg]
    
    result = {
        'lbg_table': lbg_table,
        'non_lbg_table': non_lbg_table,
        'full_table': table,  # 包含 Z_BEST, IS_LBG 列的完整表
        'stats': stats
    }
    
    if return_mask:
        masks = {
            'mask_lbg': mask_lbg,
            'mask_non_lbg': mask_non_lbg,
            'mask_vi_good': mask_vi_good,
            'mask_rr_good': mask_rr_good,
            'mask_reliable': mask_reliable
        }
        return result, masks
    
    return result


def apply_clauds_mask(
    table_or_path: Union[str, Table],
    field:  str = 'COSMOS',
    mask_col: str = 'MASK',
    flag_col: str = 'FLAG_FIELD_BINARY',
    return_mask: bool = False,
    verbose: bool = True
) -> Union[Table, Tuple[Table, np.ndarray]]:
    """
    应用 CLAUDS 测光数据的质量掩码。
    
    根据 README: 
    - COSMOS: (MASK == 0) & (FLAG_FIELD_BINARY[:,0] == True) & (FLAG_FIELD_BINARY[:,1] == True)
    - XMM:     (MASK == 0) & (FLAG_FIELD_BINARY[:,0] == True) & (FLAG_FIELD_BINARY[:,2] == True)
    
    Parameters
    ----------
    table_or_path : str or Table
        输入数据
    field : str
        天区名称:  'COSMOS' 或 'XMM'
    mask_col : str
        MASK 列名
    flag_col : str
        FLAG_FIELD_BINARY 列名
    return_mask : bool
        是否返回掩码
    verbose : bool
        是否打印信息
        
    Returns
    -------
    filtered_table : Table
        筛选后的表格
    mask : np.ndarray (optional)
        布尔掩码
    """
    
    # 加载数据
    if isinstance(table_or_path, str):
        if verbose:
            print(f"Loading: {table_or_path}")
        table = Table.read(table_or_path, hdu=1)
    else:
        table = table_or_path.copy()
    
    n_total = len(table)
    if verbose:
        print(f"Total sources: {n_total}")
    
    # 检查列名
    if mask_col not in table.colnames:
        raise ValueError(f"Column '{mask_col}' not found")
    if flag_col not in table.colnames:
        raise ValueError(f"Column '{flag_col}' not found")
    
    # 提取数据
    mask_values = np.array(table[mask_col])
    flag_values = np. array(table[flag_col])
    
    # 检查 FLAG_FIELD_BINARY 的形状
    if verbose:
        print(f"  FLAG_FIELD_BINARY shape: {flag_values.shape}")
    
    # 应用条件
    cond_mask = (mask_values == 0)
    cond_flag0 = flag_values[: , 0]. astype(bool)
    
    field_upper = field.upper()
    if field_upper == 'COSMOS':
        cond_flag_field = flag_values[:, 1].astype(bool)
    elif field_upper in ['XMM', 'XMM-LSS', 'XMMLSS']:
        cond_flag_field = flag_values[:, 2].astype(bool)
    else:
        raise ValueError(f"Unknown field: {field}. Use 'COSMOS' or 'XMM'")
    
    # 组合条件
    good_mask = cond_mask & cond_flag0 & cond_flag_field
    
    n_good = np.sum(good_mask)
    
    if verbose:
        print(f"\n  Mask conditions ({field}):")
        print(f"    - MASK == 0:              {np.sum(cond_mask):8d}")
        print(f"    - FLAG_FIELD_BINARY[:,0]:  {np.sum(cond_flag0):8d}")
        print(f"    - FLAG_FIELD_BINARY[:,{'1' if field_upper=='COSMOS' else '2'}]: {np.sum(cond_flag_field):8d}")
        print(f"    - All conditions:         {n_good:8d} ({n_good/n_total*100:.1f}%)")
    
    filtered_table = table[good_mask]
    
    if return_mask:
        return filtered_table, good_mask
    return filtered_table


def get_training_labels(
    specz_result: Dict,
    label_type: str = 'binary'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 LBG 筛选结果生成训练标签。
    
    Parameters
    ----------
    specz_result : dict
        select_lbg_from_specz 的返回结果
    label_type : str
        'binary':  返回 0/1 标签
        'redshift': 返回红移值
        
    Returns
    -------
    labels : np.ndarray
        标签数组
    z_values : np.ndarray
        红移值数组
    """
    
    full_table = specz_result['full_table']
    
    if label_type == 'binary': 
        labels = np.array(full_table['IS_LBG'])
    elif label_type == 'redshift':
        labels = np. array(full_table['Z_BEST'])
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    
    z_values = np.array(full_table['Z_BEST'])
    
    return labels, z_values


def summarize_lbg_selection(results_dict: Dict[str, Dict]) -> None:
    """
    汇总多个天区的 LBG 筛选结果。
    
    Parameters
    ----------
    results_dict : dict
        键为天区名，值为 select_lbg_from_specz 的返回结果
    """
    
    print("\n" + "=" * 70)
    print("LBG SELECTION SUMMARY (ALL FIELDS)")
    print("=" * 70)
    
    print(f"\n{'Field':<15} | {'Total': >8} | {'Reliable':>10} | {'LBG':>8} | {'Non-LBG':>10} | {'LBG %':>8}")
    print("-" * 70)
    
    total_all = 0
    lbg_all = 0
    non_lbg_all = 0
    
    for field, result in results_dict. items():
        stats = result['stats']
        total_all += stats['n_total']
        lbg_all += stats['n_lbg']
        non_lbg_all += stats['n_non_lbg']
        
        print(f"{field:<15} | {stats['n_total']:>8} | {stats['n_reliable']:>10} | "
              f"{stats['n_lbg']:>8} | {stats['n_non_lbg']:>10} | {stats['lbg_fraction']: >7.1f}%")
    
    print("-" * 70)
    lbg_pct = lbg_all / total_all * 100 if total_all > 0 else 0
    print(f"{'TOTAL':<15} | {total_all:>8} | {'-':>10} | {lbg_all:>8} | {non_lbg_all:>10} | {lbg_pct:>7.1f}%")

def get_training_samples(
    specz_result:  Dict,
    only_reliable: bool = True,
    balance_classes: bool = False,
    random_state: int = 42
) -> Dict: 
    """
    从 LBG 筛选结果生成训练样本。
    
    Parameters
    ----------
    specz_result : dict
        select_lbg_from_specz 的返回结果
    only_reliable : bool
        是否只使用可靠红移的样本 (推荐 True)
    balance_classes :  bool
        是否平衡正负样本数量 (下采样多数类)
    random_state : int
        随机种子
        
    Returns
    -------
    samples : dict
        - 'table': 训练样本表格
        - 'labels': 二分类标签 (0/1)
        - 'z_values': 红移值
        - 'stats': 统计信息
    """
    
    if only_reliable:
        # 合并 LBG 和 Non-LBG (都是可靠红移)
        from astropy.table import vstack
        
        lbg_table = specz_result['lbg_table']. copy()
        non_lbg_table = specz_result['non_lbg_table'].copy()
        
        # 确保标签正确
        lbg_table['IS_LBG'] = 1
        non_lbg_table['IS_LBG'] = 0
        
        combined_table = vstack([lbg_table, non_lbg_table])
        
    else:
        # 使用完整表格 (包含不可靠红移，不推荐)
        combined_table = specz_result['full_table'].copy()
    
    n_pos = np.sum(combined_table['IS_LBG'] == 1)
    n_neg = np.sum(combined_table['IS_LBG'] == 0)
    
    # 可选：平衡类别
    if balance_classes and n_pos != n_neg:
        np.random.seed(random_state)
        
        pos_idx = np.where(combined_table['IS_LBG'] == 1)[0]
        neg_idx = np.where(combined_table['IS_LBG'] == 0)[0]
        
        n_min = min(n_pos, n_neg)
        
        if n_pos > n_neg:
            # 下采样正样本
            sampled_pos = np.random. choice(pos_idx, n_min, replace=False)
            keep_idx = np.concatenate([sampled_pos, neg_idx])
        else:
            # 下采样负样本
            sampled_neg = np.random.choice(neg_idx, n_min, replace=False)
            keep_idx = np.concatenate([pos_idx, sampled_neg])
        
        combined_table = combined_table[keep_idx]
        n_pos = n_neg = n_min
    
    labels = np.array(combined_table['IS_LBG'])
    z_values = np.array(combined_table['Z_BEST'])
    
    stats = {
        'n_total': len(combined_table),
        'n_positive': int(n_pos),
        'n_negative': int(n_neg),
        'pos_ratio': n_pos / len(combined_table) * 100,
        'only_reliable': only_reliable,
        'balanced':  balance_classes
    }
    
    print("\n" + "=" * 50)
    print("Training Sample Summary")
    print("=" * 50)
    print(f"  Only reliable redshifts: {only_reliable}")
    print(f"  Class balancing: {balance_classes}")
    print(f"\n  Sample counts:")
    print(f"    - Positive (LBG):     {stats['n_positive']: 5d}")
    print(f"    - Negative (Non-LBG): {stats['n_negative']:5d}")
    print(f"    - Total:              {stats['n_total']:5d}")
    print(f"    - Positive ratio:     {stats['pos_ratio']:.1f}%")
    
    return {
        'table':  combined_table,
        'labels': labels,
        'z_values': z_values,
        'stats': stats
    }


def prepare_training_data_from_multiple_fields(
    results_dict: Dict[str, Dict],
    only_reliable: bool = True,
    test_field: str = None,
    verbose: bool = True
) -> Dict:
    """
    从多个天区准备训练数据，可选择留出一个天区作为测试集。
    
    Parameters
    ----------
    results_dict : dict
        键为天区名，值为 select_lbg_from_specz 的返回结果
    only_reliable : bool
        是否只使用可靠红移
    test_field : str or None
        留出作为测试集的天区名，None 表示混合所有天区
    verbose : bool
        是否打印信息
        
    Returns
    -------
    data : dict
        - 'train_table': 训练集表格
        - 'train_labels': 训练标签
        - 'test_table': 测试集表格 (如果指定了 test_field)
        - 'test_labels':  测试标签
        - 'stats': 统计信息
    """
    from astropy.table import vstack
    
    train_tables = []
    test_tables = []
    
    for field, result in results_dict.items():
        lbg_table = result['lbg_table']. copy()
        non_lbg_table = result['non_lbg_table'].copy()
        
        lbg_table['IS_LBG'] = 1
        lbg_table['FIELD'] = field
        non_lbg_table['IS_LBG'] = 0
        non_lbg_table['FIELD'] = field
        
        field_table = vstack([lbg_table, non_lbg_table])
        
        if test_field and field == test_field:
            test_tables.append(field_table)
        else:
            train_tables.append(field_table)
    
    # 合并
    train_table = vstack(train_tables) if train_tables else None
    test_table = vstack(test_tables) if test_tables else None
    
    result = {
        'train_table': train_table,
        'train_labels': np.array(train_table['IS_LBG']) if train_table else None,
        'train_z':  np.array(train_table['Z_BEST']) if train_table else None,
        'test_table': test_table,
        'test_labels': np. array(test_table['IS_LBG']) if test_table else None,
        'test_z': np.array(test_table['Z_BEST']) if test_table else None,
    }
    
    if verbose: 
        print("\n" + "=" * 60)
        print("Multi-Field Training Data Summary")
        print("=" * 60)
        
        if train_table:
            n_train = len(train_table)
            n_train_pos = np.sum(train_table['IS_LBG'] == 1)
            print(f"\n  Training set:")
            print(f"    - Total:    {n_train}")
            print(f"    - Positive: {n_train_pos} ({n_train_pos/n_train*100:.1f}%)")
            print(f"    - Negative: {n_train - n_train_pos}")
        
        if test_table: 
            n_test = len(test_table)
            n_test_pos = np.sum(test_table['IS_LBG'] == 1)
            print(f"\n  Test set ({test_field}):")
            print(f"    - Total:    {n_test}")
            print(f"    - Positive: {n_test_pos} ({n_test_pos/n_test*100:.1f}%)")
            print(f"    - Negative:  {n_test - n_test_pos}")
    
    return result
