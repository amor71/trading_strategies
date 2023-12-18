def pre_calculated(self, df):
    ## Init
    df["ema_short"] = ta.EMA({"close": df.close}, timeperiod=16)
    df["ema_long"] = ta.EMA({"close": df.close}, timeperiod=50)
    df["atr"] = ta.ATR({"low": df.low, "high": df.high, "close": df.close})
    # Find a series of 10 candles where the 20ema is at least 2 ATR below the 50ema. We'll call this Span A.
    span_a_length = int(10 * self.span_length_mult)
    span_a_atr_mult = round(2 * self.atr_mult, 2)
    df["span_a"] = (
        ((df["ema_short"] + (df["atr"] * span_a_atr_mult)) < df["ema_long"])
        .rolling(span_a_length)
        .sum()
    )
    df["final_span_a_candles"] = (df["span_a"] == span_a_length) & (
        df["span_a"].shift(-1) == span_a_length - 1
    )
    # Create a "trailing" of where span a ended for Span B to detect validity
    df["span_a_fwd_trailing_30"] = (
        df["final_span_a_candles"]
        .rolling(int(30 * self.span_length_mult))
        .max()
    )
    # find another series of 10 candles where the 20ema is
    # below the 50ema but less than 1.5 ATR below it.
    span_b_length = int(10 * self.span_length_mult)
    span_b_atr_mult = round(1.5 * self.atr_mult, 2)
    df["span_b"] = (
        (
            (df["ema_short"] < df["ema_long"])
            & (
                (df["ema_short"] + (df["atr"] * span_b_atr_mult))
                > df["ema_long"]
            )
            & (df["span_a_fwd_trailing_30"] > 0)
        )
        .rolling(span_b_length)
        .sum()
    )
    df["final_span_b_candles"] = (df["span_b"] == 10) & (
        df["span_b"].shift(-1) == 9
    )
    # X Candles only exist within +8/-8 rows of Final Span B candles
    df["span_b_rolling_ctr_8"] = (
        df["final_span_b_candles"].rolling(16, center=True).max()
    )
    df["potential_x_candles"] = df["ema_short"] >= df["ema_long"]
    df["x_candle"] = (df["potential_x_candles"] == 1) & (
        (df["final_span_b_candles"] == 1) | (df["span_b_rolling_ctr_8"] == 1)
    )
    # From each of those candles (Candle X), find the lowest low (LL) of the last 20 candles before Candle X.
    df["rolling_low_20"] = df["low"].rolling(20).min()
    df["avg_price"] = df[["open", "close"]].mean(axis=1)
    # set up peaks nulls
    df["peaks"] = np.nan
    df["peak_q"] = np.nan
    # For backtesting
    df["master_index"] = df.index
    # Rounding, for comparing ints later
    df["high"] = df["high"].round(2)
    df["open"] = df["open"].round(2)
    df["close"] = df["close"].round(2)
    df["low"] = df["low"].round(2)
    return df


def next(self, df, trade_dict):
    # trade_dict = {'trade_entry':np.nan,'entry_idx':np.nan, 'current_price':np.nan, 'master_peak_idx': np.nan}
    df = df.reset_index(drop=True)
    # Within 12 candles looking forward from that low (LL), locate a peak high. We'll call this Peak Q.
    df["rolling_fwd_high_12"] = (
        df["high"]
        .rolling(window=self.fwd_indexer, min_periods=0)
        .max()
        .round(2)
    )
    df["rolling_ctr_high_8"] = (
        df["high"].rolling(8, center=True).max().round(2)
    )
    # boolean to indicate if the candle is the peak
    df["price_peak"] = (
        (df["rolling_fwd_high_12"] == df["high"])
        & (df["rolling_ctr_high_8"] == df["high"])
        & (  # last 3 rows can't count, need price retracement
            ~df.index.isin(list(df.index[-3:]))
        )
    )
    if df["price_peak"][-40:].sum() < 1:
        return trade_dict
    df.loc[(df["price_peak"] == 1), "peaks"] = df.loc[
        (df["price_peak"] == 1), "high"
    ]
    last_peak_idx = df["peaks"].last_valid_index()
    price_peak = df.loc[last_peak_idx, "peaks"]
    # Get the price difference from the peak (Peak Q) to the lowest low (LL) and divide by 3.
    # Subtract that result from the peaks (Peak Q) price. We'll call this the floor.
    df["floor_price"] = price_peak - ((price_peak - df["rolling_low_20"]) / 3)
    # define volume peaks over a 12 candle lookforward
    # indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    # df['rolling_fwd_vol_12'] = df['volume'].rolling(window=indexer, min_periods=0).max()
    # Candle inside of Peak and Floor
    # locate an inside candle with price between the peak (Peak Q) price
    df["inside_candle"] = (df["avg_price"] > df["floor_price"]) & (
        df["avg_price"] < df["rolling_fwd_high_12"]
    )
    # Candles where an X Candle occurred followed by a price peak within 12 candles
    df["price_peak_rolling_12"] = df["price_peak"].rolling(12).max()
    df["z_candle"] = (df["x_candle"] == 1) & (
        (df["price_peak"] == 1) | (df["price_peak_rolling_12"] == 1)
    )
    # Now, locate a candle on a spike in volume that occurs between the peak (Peak Q) datetime
    # and 12 candles into the future from Peak Q, and between the peak (Peak Q) price and the floor.
    df["signal"] = (df["z_candle"] == 1) & (df["inside_candle"] == 1)
    last_signal_idx = df["signal"].last_valid_index()
    last_row = df.iloc[-1]
    if df[-20:]["signal"].sum() > 0:
        # Instantiate trade dict
        # trade_dict = {'trade_entry':np.nan,'entry_idx':np.nan, 'current_price':np.nan, 'master_peak_idx': np.nan}
        # log trade peaks
        master_peak_idx = df.loc[last_peak_idx, "master_index"]
        self.master_df.loc[master_peak_idx, "peak_q"] = self.master_df.loc[
            master_peak_idx, "high"
        ]
        # If found set a pending buy 1 ATR above the peak.
        one_atr = last_row["atr"]
        peak_price = price_peak
        if self.use_peaks == 1:
            limit_buy = round(peak_price + one_atr, 2)
        else:
            limit_buy = round(last_row["close"] + one_atr, 2)
        entry_idx = last_row.master_index  # final index
        trade_dict = {
            "trade_entry": limit_buy,
            "entry_idx": entry_idx,
            "current_price": last_row["close"],
            "master_peak_idx": master_peak_idx,
        }
    return trade_dict
