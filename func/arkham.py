import os, aiohttp, asyncio
from datetime import datetime
import pandas as pd
import pickle
from dotenv import load_dotenv

# 현재 스크립트의 디렉토리 경로를 찾습니다
script_dir = os.path.dirname(os.path.abspath(__file__))

# 스크립트 디렉토리의 상위 디렉토리에서 .env 파일을 찾습니다
env_path = os.path.join(script_dir, "../.env")

# 환경 변수를 로드합니다
load_dotenv(dotenv_path=env_path)


# 비동기 처리
async def fetch(session, url, params, headers):
    retries = 3
    for retry in range(retries):
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientPayloadError:
            if retry < retries - 1:
                print("Payload error, retrying...")
                await asyncio.sleep(1)
            else:
                print("Payload error, response payload was not completed")
                return {"error": "payload error"}
        except aiohttp.ClientResponseError as e:
            print(f"HTTP error occurred: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"error": str(e)}


async def append_arkham_data(
    start_timestamp, end_timestamp, df, from_p="", to_p=""
) -> pd.DataFrame:
    API_URL_ARKHAM = "https://api.arkhamintelligence.com/transfers"
    interval_timestamp = 15 * 60 * 1000

    if from_p and not to_p:
        df_file = f"dataframe_from_{from_p}.pkl"
        progress_file = f"progress_from_{from_p}.txt"
    elif not from_p and to_p:
        df_file = f"dataframe_to_{to_p}.pkl"
        progress_file = f"progress_to_{to_p}.txt"

    # 스크립트 시작 시 저장된 데이터 프레임 불러오기
    if os.path.exists(df_file):
        with open(df_file, "rb") as file:
            df = pickle.load(file)

    # 진행 상태 로딩
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            last_progress_index = int(file.readline().strip())
    else:
        last_progress_index = start_timestamp

    current_timestamp = last_progress_index
    processed_timestamps = set(df.index.astype(str).tolist())

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False)
    ) as session:
        tasks = []
        for i in range(current_timestamp, end_timestamp + 1, interval_timestamp):
            date_index = datetime.fromtimestamp(i / 1000).strftime(
                "%Y.%-m.%-d %H:%M"
            )  # Create date_index here to check if it's processed

            if (
                date_index in processed_timestamps
            ):  # Check if the current timestamp is already processed
                print(f"Skipping already processed timestamp: {date_index}")
                continue

            base_param = {
                "base": "binance,coinbase",
                "chains": "bitcoin",
                "tokens": "bitcoin",
                "sortDir": "asc",
                "valueGte": 5,
                "timeGte": i,
                "timeLte": i + interval_timestamp,
            }
            from_param = {"from": from_p}
            to_param = {"to": to_p}

            parameters = {**base_param}
            if from_p and not to_p:
                parameters.update(from_param)
            elif not from_p and to_p:
                parameters.update(to_param)

            headers = {"API-Key": os.getenv("ARKHAM_API_KEY")}

            # Create tasks for asynchronous execution
            tasks.append(fetch(session, API_URL_ARKHAM, parameters, headers))

        batch_size = 7  # Adjust as necessary
        for i in range(0, len(tasks), batch_size):
            results = await asyncio.gather(*tasks[i : i + batch_size])
            await asyncio.sleep(1)  # Adjust as necessary

            for j, response in enumerate(results):
                timestamp = current_timestamp + (i + j + 1) * interval_timestamp
                date_index = datetime.fromtimestamp(timestamp / 1000).strftime(
                    "%Y.%-m.%-d %H:%M"
                )

                if "transfers" in response:
                    transfers = response["transfers"]

                    total_unit_value = 0  # Reset the total unit value for each interval

                    for transfer in transfers:
                        total_unit_value += transfer["unitValue"]

                    # Adding the total_unit_value to the respective cell in the dataframe
                    if from_p and not to_p:
                        df.at[date_index, f"from_{from_p}"] = total_unit_value
                        print(f"from {total_unit_value} 추가 완료")
                    elif not from_p and to_p:
                        df.at[date_index, f"to_{to_p}"] = total_unit_value
                        print(f"to {total_unit_value} 추가 완료")
                else:
                    print(response)
                    # Handle case where there are no transfers in the response for the current timestamp
                    if from_p and not to_p:
                        df.at[date_index, f"from_{from_p}"] = 0
                        print(f"No data for timestamp: {date_index}_from")
                    elif not from_p and to_p:
                        df.at[date_index, f"to_{to_p}"] = 0
                        print(f"No data for timestamp: {date_index}_to")

                # Save the progress after processing each timestamp
                with open(df_file, "wb") as file:
                    pickle.dump(df, file)

                with open(progress_file, "w") as file:
                    file.write(str(timestamp) + "\n")
    return df
