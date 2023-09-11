import time
from func.binance import get_data
from func.rate import append_rate_data


# 데이터 처리
start_date = "2022-01-01"
end_date = "2023-09-05"
symbol = "BTCUSDT"

df = get_data(start_date, end_date, symbol)
df.to_csv(f"data/{symbol.lower()}.csv", index=False)
time.sleep(1)

df = append_rate_data(symbol=symbol)
df.to_csv("data/rate_append.csv", index=False)
time.sleep(1)
