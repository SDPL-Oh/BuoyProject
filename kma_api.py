import pandas as pd

from urllib.request import urlopen

# 해상관측 지점정보
# URL = 'https://apihub.kma.go.kr/api/typ01/url/stn_inf.php?inf=BUOY&stn=&tm=202211300900&help=1&authKey=R_zkyTnBQfy85Mk5wWH8Ow'

# 해상 부이
OCEAN_URL = 'https://apihub.kma.go.kr/api/typ01/url/kma_buoy2.php?tm1=202307231200&tm2=202307241200&stn=22183&help=1&authKey=zxiqblJaSgSYqm5SWioEeA'
FORCAST_URL = 'https://apihub.kma.go.kr/api/typ01/url/fct_medm_reg.php?tmfc=0&authKey=ImPZDKrERzmj2QyqxHc5IQ'

with urlopen(OCEAN_URL) as f:
    data = f.read().decode("cp949")
    print(data)


with urlopen(FORCAST_URL) as f:
    data = f.read().decode("cp949")
    print(data)

def save_buoy_info(raw_data, save_dir):
    lines = raw_data.strip().split('\n')
    data_lines = [line for line in lines if line and not line.strip().startswith("#") and not '----' in line]
    columns = ["STN_ID", "LON", "LAT", "STN_SP", "HT", "AD", "STN_KO", "STN_EN", "FCT_ID"]

    parsed_rows = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 7:
            stn_id = parts[0]
            lon = parts[1]
            lat = parts[2]
            stn_sp = parts[3]
            ht = parts[4]
            ad = parts[5]
            stn_ko = parts[6]
            stn_en = parts[7] if len(parts) > 7 else ""
            fct_id = parts[8] if len(parts) > 8 else ""
            parsed_rows.append([stn_id, lon, lat, stn_sp, ht, ad, stn_ko, stn_en, fct_id])


    df = pd.DataFrame(parsed_rows, columns=columns)
    df.to_csv(save_dir, index=False, encoding="utf-8-sig")


