# strategy_turtle_equity.py
"""
HS300 LightGBM 多因子策略 v3（进攻型参数）

- 价量多因子（mom20 / mom60 / mom120 / rev5 / vol20 / vol60 / liq20）
- 目标：未来 20 日收益（剪切极端值）
- 模型：LightGBM 回归，滚动面板训练
- 调仓：每 5 个交易日调仓一次，持有 TOP_N=25 只
- 风控：指数（成分股均价）60/120 日均线 + 20 日动量  → 控制整体仓位（较 v2 更激进）
- fallback：训练失败或预测退化时退回 60 日动量排序
"""

from atrader import *
import numpy as np
import os
# import lightgbm as lgb  # 旧版机器学习模块（Baseline 中未使用）


# ==================== 超参数 ====================

LABEL_HORIZON = 20          # 预测未来 20 日收益
WIN_LEN = 121               # 单次建特征窗口长度（>=121）
TRAIN_DAYS = 380            # 训练面板覆盖的交易日数（越大越稳，但更贴合样本期）
REBALANCE_GAP = 5           # 每 5 日调仓一次（更激进）
TOP_N = 25                  # 持仓股票数量（更集中）


# ==================== 工具函数 ====================

def _zscore(x):
    x = np.asarray(x, dtype='float64')
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        s = 1.0
    return (x - m) / s


def _fill_nan_2d(arr):
    """对二维数组进行简单的缺失值填补"""
    x = np.array(arr, dtype='float64')
    row_mean = np.nanmean(x, axis=1)
    global_mean = np.nanmean(x)
    row_mean = np.where(np.isnan(row_mean), global_mean, row_mean)
    idxs = np.where(np.isnan(x))
    x[idxs] = np.take(row_mean, idxs[0])
    return x


# ==================== 因子构造（保留原函数，Baseline 中不使用 ML） ====================

def build_features_window(close_win, vol_win):
    """
    在给定长度的滚动窗口上构造截面因子特征。
    close_win, vol_win: shape = (n_stocks, L)，L >= WIN_LEN
    """
    price_t = close_win[:, -1]

    # 动量因子
    mom20 = price_t / close_win[:, -21] - 1.0
    mom60 = price_t / close_win[:, -61] - 1.0
    mom120 = price_t / close_win[:, -121] - 1.0

    # 短期反转
    rev5 = -(price_t / close_win[:, -6] - 1.0)

    # 波动率
    seg20 = close_win[:, -21:]
    ret20 = seg20[:, 1:] / seg20[:, :-1] - 1.0
    vol20 = np.std(ret20, axis=1)

    seg60 = close_win[:, -61:]
    ret60 = seg60[:, 1:] / seg60[:, :-1] - 1.0
    vol60 = np.std(ret60, axis=1)

    # 流动性
    vol20_mean = np.mean(vol_win[:, -20:], axis=1)
    vol60_mean = np.mean(vol_win[:, -60:], axis=1)
    liq20 = vol20_mean / (vol60_mean + 1e-8)

    feats = np.column_stack(
        [mom20, mom60, mom120, rev5, vol20, vol60, liq20]
    )
    return feats


def build_panel_lightgbm(close, volume):
    """
    从完整 K 线面板构造 LightGBM 训练面板 + 当日特征。
    Baseline 中不再调用，但保留函数以保证结构一致。
    """
    n_stocks, T = close.shape

    # 至少要有 121 日窗口 + 20 日标签
    if T < WIN_LEN + LABEL_HORIZON:
        return None, None, None

    min_t = WIN_LEN - 1
    max_t = T - 1 - LABEL_HORIZON
    if max_t <= min_t:
        return None, None, None

    X_list, y_list = [], []

    for t in range(min_t, max_t + 1):
        close_win = close[:, t - (WIN_LEN - 1): t + 1]
        vol_win = volume[:, t - (WIN_LEN - 1): t + 1]

        feats_t = build_features_window(close_win, vol_win)
        y_t = close[:, t + LABEL_HORIZON] / close[:, t] - 1.0

        X_list.append(feats_t)
        y_list.append(y_t)

    X_panel = np.vstack(X_list)
    y_panel = np.hstack(y_list)

    # 当前截面特征
    close_win_now = close[:, -WIN_LEN:]
    vol_win_now = volume[:, -WIN_LEN:]
    X_today = build_features_window(close_win_now, vol_win_now)

    return X_panel, y_panel, X_today


# ==================== 风控：指数状态 & 仓位乘数 ====================

def compute_market_risk_mult(close):
    """
    使用成分股等权价格作为“伪指数”，
    根据 60 / 120 日均线与 20 日动量判断市场状态，输出 risk_mult。
    """
    eq_index = np.nanmean(close, axis=0)
    if len(eq_index) < 120:
        return 1.0

    ma60 = np.mean(eq_index[-60:])
    ma120 = np.mean(eq_index[-120:])
    mom20 = eq_index[-1] / eq_index[-21] - 1.0

    if eq_index[-1] >= ma60 >= ma120:
        risk_mult = 1.0
    elif eq_index[-1] >= ma120:
        risk_mult = 0.85
    else:
        risk_mult = 0.65
        if mom20 < 0:
            risk_mult = 0.45

    return float(risk_mult)


# ==================== 调仓执行 ====================

def rebalance_equal_weight(context, close_today, scores, risk_mult=1.0):
    """
    根据 scores 做等权调仓，并用 risk_mult 调整体仓位：
      - risk_mult ∈ (0,1] 决定总资金中有多少用于持股；
      - 排名前 TOP_N 的股票等权持有，其余目标仓位为 0。
    使用 order_target_volume 下单，side=1（买入方向），order_type=2（市价/限价按你原回测设定）。
    """
    n = len(scores)
    order_idx = np.argsort(-scores)
    pick_idx = order_idx[:TOP_N]
    long_set = set(pick_idx.tolist())

    acc = context.account()
    positions = acc.positions

    # 现有持仓市值 + 现金，用来算总资产
    try:
        mv_val = float(np.nansum(acc.market_value))
    except Exception:
        try:
            if hasattr(positions, 'empty') and ('volume_long' in positions.columns) and not positions.empty:
                vol_arr = positions['volume_long'].to_numpy()
                mv_val = float(np.sum(vol_arr * close_today[:len(vol_arr)]))
            else:
                mv_val = 0.0
        except Exception:
            mv_val = 0.0

    try:
        cash_val = float(np.nansum(acc.cash))
    except Exception:
        cash_val = 0.0

    total_value = mv_val + cash_val
    if not np.isfinite(total_value) or total_value <= 0:
        total_value = getattr(context, 'initial_cash', 1e8)

    # 根据市场状态调整可投资资金规模
    invest_cap = total_value * max(min(risk_mult, 1.0), 0.0)
    if invest_cap < 0:
        invest_cap = 0.0

    # 等权配置
    target_weight = 1.0 / max(len(long_set), 1)

    for i in range(n):
        price = float(close_today[i])
        if price <= 0 or np.isnan(price):
            continue

        if i in long_set and invest_cap > 0:
            target_value = invest_cap * target_weight
            # 100 股一手，向下取整
            target_vol = int(target_value / price / 100) * 100
            if target_vol < 0:
                target_vol = 0
        else:
            # 不在持仓集合里 → 目标持仓为 0
            target_vol = 0

        order_target_volume(
            account_idx=0,
            target_idx=i,
            target_volume=int(target_vol),
            side=1,
            order_type=2
        )



# ==================== ATRader 标准接口 ====================

def init(context):
    """
    初始化：
    - 注册日频 K 线（reg_kdata 结果 ATRader 自动存入 context.reg_kdata）
    - 设置回测参数
    """
    reg_kdata(frequency='day', fre_num=1)

    context.initial_cash = 1_0000_0000
    set_backtest(
        initial_cash=context.initial_cash,
        margin_rate=1.0,
        slide_price=0,
        price_loc=1,
        deal_type=0,
        limit_type=0
    )

    context.Tlen = len(context.target_list)
    context.bar = 0
    context.last_rebalance_bar = -REBALANCE_GAP


def on_data(context):
    context.bar += 1

    # 历史长度不足
    if context.bar < WIN_LEN + LABEL_HORIZON + 5:
        return

    # 控制调仓频率（与 v3 相同）
    if (context.bar - context.last_rebalance_bar) < REBALANCE_GAP:
        return

    # 取最近一段数据（保持和 v3 一样的长度逻辑）
    max_len = WIN_LEN + LABEL_HORIZON + TRAIN_DAYS + 5
    length = min(context.bar, max_len)

    data = get_reg_kdata(
        reg_idx=context.reg_kdata[0],  # ★ 用原版的注册索引，不要改
        length=length,
        fill_up=True,
        df=True
    )
    if data is None or data.empty:
        return

    close_raw = data['close'].values.reshape(context.Tlen, -1)
    vol_raw = data['volume'].values.reshape(context.Tlen, -1)

    close = _fill_nan_2d(close_raw)
    volume = _fill_nan_2d(vol_raw)

    n_stocks, T = close.shape
    if T < 121:
        # 至少 120 根K线才能算 60 日动量和 MA120
        return

    close_today = close[:, -1]

    # ===== 1. 60 日动量打分（Baseline 核心，对照组，只改这里） =====
    # mom60_i(t) = P_i(t) / P_i(t-60) - 1
    mom60_today = close_today / close[:, -61] - 1.0
    mom60_today = np.nan_to_num(mom60_today, nan=0.0)
    scores = _zscore(mom60_today)

    # ===== 2. 市场状态过滤：控制整体仓位（沿用你原来的 compute_market_risk_mult）=====
    risk_mult = compute_market_risk_mult(close)

    # ===== 3. 等权调仓（用上面改好的 rebalance_equal_weight） =====
    rebalance_equal_weight(context, close_today, scores, risk_mult=risk_mult)

    context.last_rebalance_bar = context.bar


# ==================== 本地直接回测（可选） ====================

if __name__ == '__main__':
    codes = get_code_list('hs300')['code']
    run_backtest(
        strategy_name='ML_LightGBM_HS300_v3',  # 名字你可以改成 Baseline
        file_path=os.path.basename(__file__),
        target_list=codes,
        frequency='day',
        fre_num=1,
        begin_date='2020-09-01',
        end_date='2025-09-01',
        fq=1
    )
