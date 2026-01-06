# 필요한 라이브러리 임포트
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import torch
from aurora import Batch, Metadata
import numpy as np

LAT_START = 35.19
LAT_END = 34.69
LON_START = 125.57
LON_END = 126.07

resolution = 0.25
height = int(np.abs(LAT_END - LAT_START) / resolution) + 1
width = int(np.abs(LON_END - LON_START) / resolution) + 1

# 모델 입력에 필요한 기타 설정
num_input_timesteps = 2  # 현재 시점과 이전 시점
num_pressure_levels = 1  # 해수면 예측에 초점을 맞추므로 1레벨로 설정

print(f"그리드 크기: {height} x {width}")
print(f"위도 범위: {LAT_START}° ~ {LAT_END}°")
print(f"경도 범위: {LON_START}° ~ {LON_END}°")
print("---")

# 1. 정적 변수 (Static Variables)
static_vars = {
    'lsm': torch.randn(height, width), # 육지/바다 마스크
    'z': torch.randn(height, width),   # 고도
    'slt': torch.randn(height, width)  # 토양 유형
}

# 2. 지표면 변수 (Surface-level Variables)
surf_vars = {
    '2t': torch.randn(num_input_timesteps, height, width), # 2m 기온
    '10u': torch.randn(num_input_timesteps, height, width), # 10m 동풍
    '10v': torch.randn(num_input_timesteps, height, width), # 10m 남풍
    'msl': torch.randn(num_input_timesteps, height, width)  # 해수면 기압
}

# 3. 대기 변수 (Atmospheric Variables)
atmos_vars = {
    'z': torch.randn(num_input_timesteps, num_pressure_levels, height, width),
    'u': torch.randn(num_input_timesteps, num_pressure_levels, height, width),
    'v': torch.randn(num_input_timesteps, num_pressure_levels, height, width),
    't': torch.randn(num_input_timesteps, num_pressure_levels, height, width),
    'q': torch.randn(num_input_timesteps, num_pressure_levels, height, width)
}

# 4. 메타데이터 (Metadata)
metadata = Metadata(
    lat=torch.linspace(LAT_START, LAT_END, height),
    lon=torch.linspace(LON_START, LON_END, width),
    time=(datetime.now(), datetime.now() - timedelta(hours=6)), # 현재 시점과 6시간 전 시점
    atmos_levels=torch.Tensor([1000]) # 1000 hPa 압력 레벨
)

# 5. Batch 객체 생성
sinan_batch = Batch(
    surf_vars=surf_vars,
    static_vars=static_vars,
    atmos_vars=atmos_vars,
    metadata=metadata,
)

print("\n전남 신안 해상풍력 지역의 Batch 데이터가 성공적으로 생성되었습니다.")
print("---")
print("생성된 Batch 객체의 키:")
print(sinan_batch.surf_vars.keys())
print("\n'2m 기온' 텐서의 차원:")
print(sinan_batch.surf_vars['2t'].shape)
print(sinan_batch.surf_vars['msl'].shape)


temperature_data = sinan_batch.surf_vars['2t'][0].squeeze().numpy()

# 메타데이터에서 위도와 경도 값을 가져옵니다.
lats = sinan_batch.metadata.lat.numpy()
lons = sinan_batch.metadata.lon.numpy()

plt.figure(figsize=(8, 6))
plt.imshow(temperature_data, extent=[lons.min(), lons.max(), lats.min(), lats.max()], origin='upper', cmap='viridis')

# 컬러바 추가 및 라벨을 영어로 변경
cbar = plt.colorbar()
cbar.set_label('2m Temperature (Arbitrary Unit)', rotation=270, labelpad=15)

# 축 라벨 및 제목을 영어로 변경
plt.title('2m Temperature Distribution in Sinan Offshore Wind Farm Area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 그리드 표시
plt.grid(True)

# 그래프 출력
plt.show()