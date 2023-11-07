from func.binance import get_data
from func.rate import get_rate_data
from func.arkham import get_arkham_data

start_date = "2022-09-06"
end_date = "2023-10-30"
symbol = "BTCUSDT"

df = get_data(start_date, end_date, symbol)
