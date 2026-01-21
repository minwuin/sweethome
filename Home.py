import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN 
from utils.map_utils import draw_map
from utils.data_loader import (
    get_real_estate_data, 
    get_cctv_data, 
    get_noise_data, 
    get_convenience_data, 
    get_store_data
)

def calculate_distance(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

st.set_page_config(layout="wide", page_title="SweetHome - 영남대 원룸")
st.title("📍 SweetHome: 영남대 자취방 지도 (블록 분석)")

FIXED_BOUNDS = {
    'min_lat': 35.835510, 'max_lat': 35.842292,
    'min_lon': 128.750314, 'max_lon': 128.760809
}

# 1. 데이터 로드
with st.spinner("주변 시설 데이터를 불러오는 중입니다..."):
    df_price = get_real_estate_data()
    cctv_df = get_cctv_data()
    
    # [수정] CCTV 데이터 범위 제한 (범위 밖 데이터 즉시 제거)
    if not cctv_df.empty:
        cctv_df = cctv_df[
            (cctv_df['lat'] >= FIXED_BOUNDS['min_lat']) & (cctv_df['lat'] <= FIXED_BOUNDS['max_lat']) &
            (cctv_df['lon'] >= FIXED_BOUNDS['min_lon']) & (cctv_df['lon'] <= FIXED_BOUNDS['max_lon'])
        ]
    
    noise_df = get_noise_data(**FIXED_BOUNDS)       
    convenience_df = get_convenience_data(**FIXED_BOUNDS) 
    store_df = get_store_data(**FIXED_BOUNDS)       

BUILD_PATH = r"C:\minwoin\room\data\buildings.csv"
if not os.path.exists(BUILD_PATH):
    st.error("❌ buildings.csv가 없습니다.")
    st.stop()
df_build = pd.read_csv(BUILD_PATH)

# 2. 사이드바 (슬라이더 제거 및 고정값 적용)
# 2. 사이드바
with st.sidebar:
    # [추가] 가격 필터 슬라이더
    st.header("🔍 필터 설정")
    with st.expander("💰 가격 조건 (블록 평균)", expanded=True):
        deposit_range = st.slider(
            "평균 보증금 (만원)", 
            min_value=50, max_value=2000, 
            value=(50, 2000), step=10
        )
        rent_range = st.slider(
            "평균 월세 (만원)", 
            min_value=20, max_value=100, 
            value=(20, 100), step=5
        )
    
    st.divider()
    st.subheader("시설 표시")
    # ... (이하 show_cctv 등 기존 코드 유지)
    st.divider()
    st.subheader("시설 표시")
    show_cctv = st.toggle("CCTV (🎥)", value=True)
    show_conv = st.toggle("편의점 (🛒)", value=True)
    show_noise = st.toggle("소음원 (🍺/🎵)", value=False)
    show_store = st.toggle("상가 (🍴)", value=False)
    
    st.divider()
    st.caption(f"📊 분석 대상 건물: {len(df_build)}개")

# 3. 데이터 전처리
df_build['노후도'] = pd.to_numeric(df_build['노후도'], errors='coerce').fillna(0)
df_build['lat'] = pd.to_numeric(df_build['lat'], errors='coerce')
df_build['lon'] = pd.to_numeric(df_build['lon'], errors='coerce')

df_build['법정동_정제'] = df_build['법정동'].astype(str).apply(lambda x: x.split()[-1].strip())
df_price['법정동_정제'] = df_price['법정동'].astype(str).apply(lambda x: x.split()[-1].strip())
df_price['보증금'] = pd.to_numeric(df_price['보증금'], errors='coerce').fillna(0)
df_price['월세'] = pd.to_numeric(df_price['월세'], errors='coerce').fillna(0)

price_stats = df_price.groupby('법정동_정제')[['보증금', '월세']].mean().reset_index()
merged_df = pd.merge(df_build, price_stats, on='법정동_정제', how='left').fillna(0)

merged_df = merged_df[
    (merged_df['lat'] >= FIXED_BOUNDS['min_lat']) & (merged_df['lat'] <= FIXED_BOUNDS['max_lat']) &
    (merged_df['lon'] >= FIXED_BOUNDS['min_lon']) & (merged_df['lon'] <= FIXED_BOUNDS['max_lon'])
].copy()

# 4. DBSCAN 군집화
if len(merged_df) > 0:
    block_eps = 17
    block_min = 3
    coords = np.radians(merged_df[['lat', 'lon']].values)
    kms_per_radian = 6371.0088
    epsilon = (block_eps / 1000) / kms_per_radian
    
    db = DBSCAN(eps=epsilon, min_samples=block_min, metric='haversine', algorithm='ball_tree').fit(coords)
    merged_df['cluster'] = db.labels_
    
    clustered_df = merged_df[merged_df['cluster'] != -1].copy()
    
    block_stats = clustered_df.groupby('cluster').agg({
        'lat': 'mean',
        'lon': 'mean',
        '월세': 'mean',
        '보증금': 'mean',
        '건물명': 'count'
    }).reset_index()
    
    def count_nearby(center_lat, center_lon, target_df, radius=100):
        if target_df.empty: return 0
        dists = calculate_distance(center_lat, center_lon, target_df['lat'].values, target_df['lon'].values)
        return np.sum(dists <= radius)

    block_stats['cctv_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], cctv_df), axis=1)
    block_stats['conv_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], convenience_df), axis=1)

    # ---------------------------------------------------------
    # [여기에 추가] 사용자가 설정한 슬라이더 값으로 블록 필터링
    # ---------------------------------------------------------
    # 1. 블록별 평균 가격이 슬라이더 범위 내에 있는지 확인
    filtered_block_stats = block_stats[
        (block_stats['보증금'] >= deposit_range[0]) & (block_stats['보증금'] <= deposit_range[1]) &
        (block_stats['월세'] >= rent_range[0]) & (block_stats['월세'] <= rent_range[1])
    ]
    
    # 2. 필터링된 블록들의 cluster ID 목록을 가져옴
    valid_cluster_ids = filtered_block_stats['cluster'].tolist()
    
    # 3. 지도에 표시할 개별 건물 데이터도 해당 블록 ID만 남김
    filtered_clustered_df = clustered_df[clustered_df['cluster'].isin(valid_cluster_ids)]

else:
    clustered_df = pd.DataFrame()
    block_stats = pd.DataFrame()
    # 데이터가 없을 경우를 대비해 필터링 변수도 초기화
    filtered_block_stats = pd.DataFrame()
    filtered_clustered_df = pd.DataFrame()

# 5. 지도 그리기
final_cctv = cctv_df if show_cctv else pd.DataFrame()
final_noise = noise_df if show_noise else pd.DataFrame()
final_conv = convenience_df if show_conv else pd.DataFrame()
final_store = store_df if show_store else pd.DataFrame()

# [수정] filtered_block_stats를 기준으로 체크
if len(filtered_block_stats) > 0:
    st.success(f"📍 조건에 맞는 블록을 **{len(filtered_block_stats)}개** 찾았습니다.")
    
    # [수정] draw_map에 필터링된 데이터 전달
    m = draw_map(
        filtered_clustered_df, 
        filtered_block_stats, 
        final_cctv, 
        final_noise, 
        final_conv, 
        final_store
    )
    
    if m:
        st_folium(m, width="100%", height=600)
    else:
        st.error("지도 생성 실패")
else:
    st.warning("선택하신 가격 조건에 맞는 블록이 없습니다.")