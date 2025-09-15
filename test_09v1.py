# app_preprocess_robust_fast_map_v5_integrated.py
# 실행: streamlit run app_preprocess_robust_fast_map_v5_integrated.py
# 필요: pip install streamlit pandas numpy folium streamlit-folium

import streamlit as st
import pandas as pd
import numpy as np
import io, csv, re, math

import folium
from folium.plugins import TimestampedGeoJson
from streamlit_folium import st_folium

# ---------------- 기본 설정 ----------------
st.set_page_config(page_title="전처리(견고+고속)+지도(이동/정지 구분)", layout="wide")
st.title("위치데이터 테스트")

BASE_NCOL = 20  # A..T 헤더 기준

# 원본형(H/I/J/K/N → 0-based 7/8/9/10/13), 첨부형(J/K/L/M/P → 0-based 9/10/11/12/15)
POS_ORIG = dict(time=7,  lat=8,  lon=9,  alt=10, sats=13)  # H I J K N
POS_CURR = dict(time=9,  lat=10, lon=11, alt=12, sats=15)  # J K L M P

# 현재 첨부 파일에서 자주 보이는 컬럼명(예: 0915_위성망_RTK_JSON.csv)
NAME_CURR = {
    "time": ["timeUtc", "timestamp", "QF_TIMESTAMP_ISO8601", "QF_COLLECTOR_TIMESTAMP_ISO8601"],
    "lat":  ["latitudeDegree", "lat", "Latitude"],
    "lon":  ["longitudeDegree", "lon", "Longitude"],
    "alt":  ["altitudeMeter", "altitude", "alt", "Altitude"],
    "sats": ["satellites", "sats", "Satellites"]
}

ISO_RE = re.compile(r'(?P<iso>\d{4}-\d{2}-\d{2}T[0-9:\.]+Z)')
GPGGA_RE = re.compile(
    r'\$GPGGA,'
    r'(?P<hms>\d{5,6}(?:\.\d+)?)\s*,'
    r'(?P<lat>\d{2,4}\.\d+)\s*,(?P<ns>[NS])\s*,'
    r'(?P<lon>\d{3,5}\.\d+)\s*,(?P<ew>[EW])\s*,'
    r'(?P<fix>[0-6])\s*,'
    r'(?P<sats>\d{1,2})\s*,'
    r'(?P<hdop>\d+(?:\.\d+)?)\s*,'
    r'(?P<alt>-?\d+(?:\.\d+)?)\s*,M,'
    r'(?P<geoid>-?\d+(?:\.\d+)?)\s*,M',
    re.IGNORECASE
)

# ---------------- 타임존 유틸 ----------------
def get_zoneinfo(name: str):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        try:
            import pytz
            return pytz.timezone(name)
        except Exception:
            return None

def to_local_strings(ts_utc: pd.Series, tz_name: str) -> pd.Series:
    tz = get_zoneinfo(tz_name)
    s = pd.to_datetime(ts_utc, utc=True)
    if tz is None:
        return s.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return s.dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S")

def time_for_map_strings(ts_utc: pd.Series, tz_name: str) -> pd.Series:
    """Folium 타임슬라이더 표기를 Streamlit 표기와 맞추기 위해 로컬시간 문자열로 전달"""
    tz = get_zoneinfo(tz_name)
    s = pd.to_datetime(ts_utc, utc=True)
    if tz is None:
        return s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # UTC
    local = s.dt.tz_convert(tz)
    return local.dt.strftime("%Y-%m-%dT%H:%M:%S")  # 오프셋 없이 로컬 표기

def iso_no_ms(s: pd.Series) -> pd.Series:
    """UTC 기준 초단위 ISO 문자열(원본 형식과 호환)"""
    ss = pd.to_datetime(s, utc=True, errors="coerce")
    return ss.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------------- 로더(견고/고속) ----------------
def read_robust_from_bytes(raw: bytes) -> pd.DataFrame:
    # 1) C엔진(빠름)
    try:
        df = pd.read_csv(io.BytesIO(raw), engine="c", sep=",", dtype=str, quoting=csv.QUOTE_MINIMAL)
        if df.shape[0] > 0:
            return df
    except Exception:
        pass
    # 2) python 엔진(따옴표 지원)
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr","latin1"):
        try:
            buf = io.StringIO(raw.decode(enc, errors="replace"))
            df = pd.read_csv(buf, engine="python", sep=None, quotechar='"', doublequote=True,
                             on_bad_lines="skip", dtype=str)
            if df.shape[0] > 0:
                return df
        except Exception:
            pass
    # 3) QUOTE_NONE 폴백
    best, best_score = None, (-1, -1)
    for enc in ("utf-8-sig","utf-8","cp949","euc-kr","latin1"):
        try:
            buf = io.StringIO(raw.decode(enc, errors="replace"))
            df = pd.read_csv(buf, engine="python", sep=None, quoting=csv.QUOTE_NONE,
                             escapechar="\\", on_bad_lines="skip", dtype=str)
            score = (df.shape[1], df.shape[0])
            if score > best_score:
                best, best_score = df, score
        except Exception:
            pass
    if best is not None:
        return best
    raise ValueError("파일 파싱 실패")

# ---------------- 좌표 변환 ----------------
def ddmm_to_deg(v):
    try:
        v = float(v)
        d = int(v // 100); mi = v - d*100
        return d + mi/60.0
    except Exception:
        return np.nan

# ---------------- 유틸: 후보 컬럼 찾기 ----------------
def find_first_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------------- 고속 경로: 두 형식 자동 인식 ----------------
def fast_flexible_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    두 형식(A: 기존 원본 H/I/J/K/N, B: 현재 첨부 J/K/L/M/P 또는 이름 기반)을 자동 인식해서
    공통 출력 스키마(timeUtc, lat, lon, alt, sats)를 반환.
    ddmm → deg 보정 및 범위/NaN 필터 포함.
    """
    candidates = []

    # 0) 이름 기반(현재 첨부형 가능)
    tcol = find_first_col(df, NAME_CURR["time"])
    la = find_first_col(df, NAME_CURR["lat"])
    lo = find_first_col(df, NAME_CURR["lon"])
    al = find_first_col(df, NAME_CURR["alt"])
    sa = find_first_col(df, NAME_CURR["sats"])
    if all([tcol, la, lo, al, sa]):
        try:
            s_time = pd.to_datetime(df[tcol], errors="coerce", utc=True)
            s_lat  = pd.to_numeric(df[la], errors="coerce")
            s_lon  = pd.to_numeric(df[lo], errors="coerce")
            s_alt  = pd.to_numeric(df[al], errors="coerce")
            s_sats = pd.to_numeric(df[sa], errors="coerce")
            # ddmm 패턴 보정
            if (s_lat > 90).mean() > 0.3:  s_lat = df[la].map(ddmm_to_deg)
            if (s_lon > 180).mean() > 0.3: s_lon = df[lo].map(ddmm_to_deg)
            out = pd.DataFrame({"timeUtc": s_time, "lat": s_lat, "lon": s_lon, "alt": s_alt, "sats": s_sats})
            out = out.dropna(subset=["timeUtc","lat","lon"])
            out = out[(out["lat"].between(-90,90)) & (out["lon"].between(-180,180))]
            candidates.append(("name_curr", out))
        except Exception:
            pass

    # 1) 위치 기반 — 기존 원본(H/I/J/K/N)
    if df.shape[1] >= BASE_NCOL:
        try:
            s_time = pd.to_datetime(df.iloc[:, POS_ORIG["time"]], errors="coerce", utc=True)
            s_lat  = pd.to_numeric(df.iloc[:, POS_ORIG["lat"]],  errors="coerce")
            s_lon  = pd.to_numeric(df.iloc[:, POS_ORIG["lon"]],  errors="coerce")
            s_alt  = pd.to_numeric(df.iloc[:, POS_ORIG["alt"]],  errors="coerce")
            s_sats = pd.to_numeric(df.iloc[:, POS_ORIG["sats"]], errors="coerce")
            if (s_lat > 90).mean() > 0.3:  s_lat = df.iloc[:, POS_ORIG["lat"]].map(ddmm_to_deg)
            if (s_lon > 180).mean() > 0.3: s_lon = df.iloc[:, POS_ORIG["lon"]].map(ddmm_to_deg)
            out = pd.DataFrame({"timeUtc": s_time, "lat": s_lat, "lon": s_lon, "alt": s_alt, "sats": s_sats})
            out = out.dropna(subset=["timeUtc","lat","lon"])
            out = out[(out["lat"].between(-90,90)) & (out["lon"].between(-180,180))]
            candidates.append(("pos_orig", out))
        except Exception:
            pass

    # 2) 위치 기반 — 현재 첨부(J/K/L/M/P)
    if df.shape[1] >= max(POS_CURR.values())+1:
        try:
            s_time = pd.to_datetime(df.iloc[:, POS_CURR["time"]], errors="coerce", utc=True)
            s_lat  = pd.to_numeric(df.iloc[:, POS_CURR["lat"]],  errors="coerce")
            s_lon  = pd.to_numeric(df.iloc[:, POS_CURR["lon"]],  errors="coerce")
            s_alt  = pd.to_numeric(df.iloc[:, POS_CURR["alt"]],  errors="coerce")
            s_sats = pd.to_numeric(df.iloc[:, POS_CURR["sats"]], errors="coerce")
            if (s_lat > 90).mean() > 0.3:  s_lat = df.iloc[:, POS_CURR["lat"]].map(ddmm_to_deg)
            if (s_lon > 180).mean() > 0.3: s_lon = df.iloc[:, POS_CURR["lon"]].map(ddmm_to_deg)
            out = pd.DataFrame({"timeUtc": s_time, "lat": s_lat, "lon": s_lon, "alt": s_alt, "sats": s_sats})
            out = out.dropna(subset=["timeUtc","lat","lon"])
            out = out[(out["lat"].between(-90,90)) & (out["lon"].between(-180,180))]
            candidates.append(("pos_curr", out))
        except Exception:
            pass

    if not candidates:
        return pd.DataFrame()

    # 가장 많은 유효행을 가진 결과를 채택
    label, best = max(candidates, key=lambda kv: kv[1].shape[0])
    return best

# ---------------- 폴백: NMEA/ISO 추출 ----------------
def robust_from_nmea(df: pd.DataFrame, file_text: str) -> pd.DataFrame:
    row_text = df.astype(str).agg(",".join, axis=1)

    g = row_text.str.extract(GPGGA_RE)
    lat = g["lat"].map(ddmm_to_deg)
    lon = g["lon"].map(ddmm_to_deg)
    lat = np.where(g["ns"].str.upper()=="S", -lat, lat)
    lon = np.where(g["ew"].str.upper()=="W", -lon, lon)
    alt  = pd.to_numeric(g["alt"],  errors="coerce")
    sats = pd.to_numeric(g["sats"], errors="coerce")

    # ✅ str.extract 사용
    iso = row_text.str.extract(ISO_RE)
    iso_series = iso["iso"] if "iso" in iso else pd.Series(index=row_text.index, dtype=str)

    base = re.search(r'(\d{4}-\d{2}-\d{2})T', file_text)
    base_date = base.group(1) if base else None

    def hms_to_iso(hms):
        if not isinstance(hms, str) or not hms: return None
        hh = int(hms[0:2]); mm = int(hms[2:4]); ssf = float(hms[4:])
        s = int(ssf); ms = int(round((ssf - s)*1000))
        return f"{hh:02d}:{mm:02d}:{s:02d}.{ms:03d}Z"

    hms_iso = g["hms"].map(hms_to_iso)
    if base_date:
        fb = base_date + "T" + hms_iso.fillna("").astype(str)
        fb_time = pd.to_datetime(fb.where(hms_iso.notna(), None), errors="coerce", utc=True)
    else:
        fb_time = pd.Series(pd.NaT, index=row_text.index)

    timeUtc = pd.to_datetime(iso_series, errors="coerce", utc=True).fillna(fb_time)

    out = pd.DataFrame({"timeUtc": timeUtc, "lat": pd.to_numeric(lat, errors="coerce"),
                        "lon": pd.to_numeric(lon, errors="coerce"), "alt": alt, "sats": sats})
    out = out.dropna(subset=["timeUtc","lat","lon"])
    out = out[(out["lat"].between(-90,90)) & (out["lon"].between(-180,180))]
    return out

# ---------------- 10Hz → 1Hz ----------------
def to_1hz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timeUtc", kind="stable")
    sec = df["timeUtc"].dt.floor("S")
    return df.loc[~sec.duplicated(), ["timeUtc","lat","lon","alt","sats"]].copy()

# ---------------- tuple 안전 래퍼 ----------------
def as_float_array(x) -> np.ndarray:
    """
    Series/list/tuple/ndarray 등 어떤 입력이 오더라도
    확실히 float 1D ndarray로 변환한다.
    """
    a = np.asarray(x, dtype=object)
    if a.ndim == 0:
        return np.array([], dtype=float)
    if a.dtype == object and len(a) and isinstance(a.flat[0], (tuple, list, np.ndarray)):
        a = np.array([v[0] if isinstance(v, (tuple, list, np.ndarray)) else v for v in a], dtype=float)
    else:
        a = a.astype(float, copy=False)
    return a

# ---------------- 거리/이동/정지 ----------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(as_float_array(lat1))
    lon1 = np.radians(as_float_array(lon1))
    lat2 = np.radians(as_float_array(lat2))
    lon2 = np.radians(as_float_array(lon2))
    if lat1.size == 0 or lat2.size == 0:
        return np.array([], dtype=float)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def split_moving_static(df_1hz: pd.DataFrame, eps_m: float = 1.0):
    """
    이동 시작 지점 이후의 데이터만 대상으로 이동/정지 포인트를 분리하고
    연속 정지 구간을 요약한다. (인덱스는 RangeIndex로 강제)
    """
    if df_1hz.shape[0] < 2:
        empty_seg = pd.DataFrame(columns=["start_time","end_time","duration_s","lat","lon","count"])
        return df_1hz.copy(), df_1hz.iloc[0:0].copy(), empty_seg

    df1 = df_1hz.reset_index(drop=True).copy()

    lat = as_float_array(df1["lat"].to_numpy())
    lon = as_float_array(df1["lon"].to_numpy())

    dist = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dist = np.r_[0.0, dist]  # 첫 시점 0m

    moving_mask = dist > eps_m

    idx0 = int(np.argmax(moving_mask)) if moving_mask.any() else 0
    df_after = df1.iloc[idx0:].reset_index(drop=True)
    moving_after = moving_mask[idx0:]

    moving_points = df_after.loc[moving_after].copy()
    static_points = df_after.loc[~moving_after].copy()

    if static_points.empty:
        static_segments = pd.DataFrame(columns=["start_time","end_time","duration_s","lat","lon","count"])
    else:
        # 부울 변화로 그룹핑(인덱스 산술 X → tuple - tuple 문제 원천 차단)
        s_mask = pd.Series(~moving_after)  # 정지=True
        grp_id = s_mask.ne(s_mask.shift(1, fill_value=False)).cumsum()
        static_points = static_points.assign(_gid=grp_id[s_mask].to_numpy())
        static_segments = (
            static_points
            .groupby("_gid", as_index=False)
            .agg(
                start_time=("timeUtc","first"),
                end_time=("timeUtc","last"),
                lat=("lat","mean"),
                lon=("lon","mean"),
                count=("timeUtc","count"),
            )
            .drop(columns="_gid")
        )
        static_segments["duration_s"] = (
            static_segments["end_time"] - static_segments["start_time"]
        ).dt.total_seconds().astype(int)
        static_segments = static_segments[["start_time","end_time","duration_s","lat","lon","count"]]

    return moving_points, static_points, static_segments

# ---------------- 지도 ----------------
def add_plane_marker(m, lat, lon, tooltip):
    try:
        folium.Marker(
            location=[float(lat), float(lon)],
            tooltip=tooltip,
            icon=folium.Icon(icon="plane", prefix="fa")
        ).add_to(m)
    except Exception:
        folium.Marker(location=[float(lat), float(lon)], tooltip=tooltip).add_to(m)

def render_map(moving_points: pd.DataFrame, full_df_1hz: pd.DataFrame, tz_name: str, speed: int = 10):
    if full_df_1hz.empty:
        st.warning("지도에 표시할 데이터가 없습니다.")
        return

    poly_src = moving_points if not moving_points.empty else full_df_1hz
    center_lat = float(poly_src["lat"].median()); center_lon = float(poly_src["lon"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, control_scale=True)

    # 전체 경로: 약간 두꺼운 파란색 라인
    coords = poly_src[["lat","lon"]].to_numpy().tolist()
    if coords:
        folium.PolyLine(
            locations=coords,
            weight=5,
            opacity=0.9,
            color="#1E90FF"  # DodgerBlue
        ).add_to(m)

    # 시작/끝 마커(비행기 아이콘)
    first = poly_src.iloc[0]
    last  = poly_src.iloc[-1]
    add_plane_marker(m, first["lat"], first["lon"], tooltip="Start")
    add_plane_marker(m, last["lat"], last["lon"], tooltip="End")

    # 애니메이션용 데이터 준비
    anim_src = moving_points if not moving_points.empty else full_df_1hz
    anim_df = anim_src.copy()
    anim_df["_time_for_map"] = time_for_map_strings(anim_df["timeUtc"], tz_name)
    anim_df["_popup_local"]  = to_local_strings(anim_df["timeUtc"], tz_name)

    # 빨간 역삼각형 SVG 아이콘 (Leaflet L.Icon 옵션으로 사용)
    TRI_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">'
        '<polygon points="10,18 2,4 18,4" fill="red"/></svg>'
    )
    iconstyle = {
        "iconUrl": "data:image/svg+xml;utf8," + TRI_SVG.replace("#","%23").replace("\n",""),
        "iconSize": [20, 20],
        "iconAnchor": [10, 18],   # 삼각형 꼭짓점이 위치를 가리키도록
        "popupAnchor": [0, -18],
    }

    # 각 시점 Feature 생성
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
        "properties": {
            "time": r["_time_for_map"],
            "popup": (
                f"time: {r['_popup_local']} ({tz_name})"
                f"<br>alt: {'' if pd.isna(r['alt']) else r['alt']}"
                f"<br>sats: {'' if pd.isna(r['sats']) else r['sats']}"
            ),
            "icon": "marker",       # L.Marker 사용
            "iconstyle": iconstyle  # 위에서 정의한 SVG 아이콘 사용
        }
    } for _, r in anim_df.iterrows()]

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT1S",
        duration="PT1S",          # 포인트는 1초만 유지 → 항상 현재 1개만
        add_last_point=True,     # 마지막 포인트 고정 마커 비활성화
        auto_play=True,
        loop=True,
        max_speed=int(speed),
        loop_button=True,
        date_options="YYYY-MM-DD HH:mm:ss",
        time_slider_drag_update=False,
        transition_time=0
    ).add_to(m)

    st_folium(m, width=None, height=560)


# ---------------- 원본 형식(A..T, H/I/J/K/N 자리) CSV 만들기 ----------------
def to_original_layout_20(df_1hz: pd.DataFrame, template_cols: list[str] | None = None) -> pd.DataFrame:
    """
    df_1hz(timeUtc, lat, lon, alt, sats)을 원본 포맷(A..T 20열, H/I/J/K/N 위치)으로 재배치.
    template_cols가 주어지면 그 헤더를 사용(가능하면 원본 헤더 유지), 없으면 A..T 생성.
    """
    n = BASE_NCOL  # 20
    if template_cols is not None and len(template_cols) >= n:
        cols = list(template_cols[:n])
    else:
        cols = [chr(ord('A')+i) for i in range(n)]

    out = pd.DataFrame(index=range(len(df_1hz)), columns=cols, dtype=object)
    out.iloc[:, POS_ORIG["time"]] = iso_no_ms(df_1hz["timeUtc"])
    out.iloc[:, POS_ORIG["lat"]]  = df_1hz["lat"].values
    out.iloc[:, POS_ORIG["lon"]]  = df_1hz["lon"].values
    out.iloc[:, POS_ORIG["alt"]]  = df_1hz["alt"].values
    out.iloc[:, POS_ORIG["sats"]] = df_1hz["sats"].values
    return out

# ---------------- UI ----------------
file = st.file_uploader("원본/현재 첨부 형식 파일 업로드 (.csv/.xlsx/.xls)", type=["csv","xlsx","xls"])
col1, col2, col3 = st.columns([1.2,1,1])
with col1:
    tz_name = st.selectbox("표시 타임존", ["Asia/Seoul","UTC"], index=0)
with col2:
    speed = st.slider("애니메이션 속도(max_speed)", 1, 20, 10)
with col3:
    eps_m = st.number_input("정지 판단 임계(m)", min_value=0.0, value=1.0, step=0.5)

if not file:
    st.info("CSV가 깨져도 자동 복원됩니다. 업로드하면 처리합니다.")
    st.stop()

# 읽기
try:
    raw = file.read()
    df_raw = read_robust_from_bytes(raw)
    file_text = raw.decode(errors="replace")
except Exception as e:
    st.error(f"파일 읽기 오류: {e}")
    st.stop()

# 1) 고속 경로(두 형식 자동 인식) → 2) 폴백(NMEA)
core_fast = fast_flexible_columns(df_raw)
core = core_fast if not core_fast.empty else robust_from_nmea(df_raw, file_text)
if core.empty:
    st.error("전처리 결과가 비었습니다. (타임스탬프/좌표 복원 실패)")
    st.stop()

# 1Hz + RangeIndex 강제
df_1hz = to_1hz(core).reset_index(drop=True)

# 이동/정지 분리 (이동 시작 이후)
moving_points, static_points, static_segments = split_moving_static(df_1hz, eps_m=eps_m)

# 로컬시간 컬럼(스트림릿 표기용)
df_1hz_display = df_1hz.copy()
df_1hz_display["time_local"] = to_local_strings(df_1hz_display["timeUtc"], tz_name)

# 요약 & 다운로드
left, right = st.columns([2,1])
with left:
    st.success(f"원본 {df_raw.shape} → core {core.shape} → 1Hz {df_1hz.shape} | "
               f"이동포인트 {moving_points.shape} | 정지포인트 {static_points.shape}")
with right:
    st.download_button(
        "최종 1Hz CSV (분석용 5열)",
        data=df_1hz.to_csv(index=False).encode("utf-8-sig"),
        file_name="preprocessed_core_1hz_5cols.csv",
        mime="text/csv"
    )

# --- 원본 형식 CSV (A..T, H/I/J/K/N에 채움) ---
orig_headers = list(df_raw.columns[:BASE_NCOL]) if df_raw.shape[1] >= BASE_NCOL else None
df_1hz_origfmt = to_original_layout_20(df_1hz, template_cols=orig_headers)

st.download_button(
    "최종 1Hz CSV (원본 형식 H/I/J/K/N)",
    data=df_1hz_origfmt.to_csv(index=False).encode("utf-8-sig"),
    file_name="preprocessed_core_1hz_original_format.csv",
    mime="text/csv"
)

st.subheader("시간에 따른 이동 경로 (이동 포인트 기준)")
render_map(moving_points, df_1hz, tz_name, speed=speed)

with st.expander("미리보기 — 최종 1Hz(상위 20행, 로컬시간 포함)"):
    st.dataframe(df_1hz_display.head(20), use_container_width=True)
with st.expander("정지 세그먼트 요약"):
    st.dataframe(static_segments, use_container_width=True)
