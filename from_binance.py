from datetime import datetime
from func.arkham import append_arkham_data
import pandas as pd
import asyncio


async def from_binance():
    start_datetime = datetime(2022, 1, 1, 00, 00, 00)
    start_timestamp = int(start_datetime.timestamp()) * 1000
    end_datetime = datetime(2023, 9, 5, 23, 45, 00)
    end_timestamp = int(end_datetime.timestamp()) * 1000

    BINANCE = "binance"

    df_rate = pd.read_csv("data/rate_append.csv", index_col=0, parse_dates=True)
    data_fromB = await append_arkham_data(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        df=df_rate,
        from_p=BINANCE,
    )
    data_fromB.to_csv(f"data/from_{BINANCE}.csv", index=True)


asyncio.run(from_binance())
