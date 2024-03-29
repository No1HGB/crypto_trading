import requests
from datetime import datetime
import time
import pandas as pd


def get_data(start_date, end_date, symbol) -> pd.DataFrame:
    COLUMNS = [
        "Open_time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close_time",
        "quote_av",
        "trades",
        "tb_base_av",
        "tb_quote_av",
        "ignore",
    ]
    URL = "https://api.binance.com/api/v3/klines"
    data = []

    start = (
        int(
            time.mktime(
                datetime.strptime(start_date + " 00:00", "%Y-%m-%d %H:%M").timetuple()
            )
        )
        * 1000
    )
    end = (
        int(
            time.mktime(
                datetime.strptime(end_date + " 23:59", "%Y-%m-%d %H:%M").timetuple()
            )
        )
        * 1000
    )
    params = {
        "symbol": symbol,
        "interval": "15m",
        "limit": 1000,
        "startTime": start,
        "endTime": end,
    }

    while start < end:
        print(datetime.fromtimestamp(start // 1000))
        params["startTime"] = start
        result = requests.get(URL, params=params)
        js = result.json()
        if not js:
            break
        data.extend(js)  # result에 저장
        start = js[-1][0] + 60000  # 다음 step으로
    # 전처리
    if not data:  # 해당 기간에 데이터가 없는 경우
        print("해당 기간에 일치하는 데이터가 없습니다.")
        return -1
    df = pd.DataFrame(data)
    df.columns = COLUMNS
    df["Open_time"] = df.apply(
        lambda x: datetime.fromtimestamp(x["Open_time"] // 1000), axis=1
    )
    df = df.drop(columns=["Close_time", "ignore"])
    df[
        df.columns[df.columns.get_loc("Open") : df.columns.get_loc("tb_quote_av") + 1]
    ] = df.loc[:, "Open":"tb_quote_av"].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df
