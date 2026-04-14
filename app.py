import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import os
import time
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np
import math
import pydeck as pdk
import base64


#hi
# ----------------  PREMIUM LIGHT UI ----------------
st.markdown("""
<style>

/* 🌤 Background */
html, body, [class*="css"] {
    background: linear-gradient(180deg, #f8fafc, #eef2ff);
    color: #111827;
}

/* 🧊 Glass cards */
.card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}

/* 🔘 Buttons */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}

.stButton button:hover {
    transform: scale(1.03);
}

/* 📊 Metric cleanup */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)





# ---------------- LOAD MODEL ----------------
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(base_path, "columns.pkl"), "rb"))

# ---------------- SESSION ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ---------------- PAGE ----------------
st.set_page_config(page_title="Smart Delivery System", layout="wide")
st.title("🚚 Delivery Routing System")

left, right = st.columns([3, 1])

# ---------------- INPUT ----------------
with right:
    st.subheader("📦 Setup")

    pickup_lat = st.number_input("Warehouse Latitude", value=28.6)
    pickup_lon = st.number_input("Warehouse Longitude", value=77.2)

    num_stops = st.number_input("Stops", 1, 5, 2)

    delivery_points = []
    for i in range(num_stops):
        lat = st.number_input(f"Lat {i+1}", key=f"lat{i}")
        lon = st.number_input(f"Lon {i+1}", key=f"lon{i}")
        delivery_points.append((lat, lon))

    weather = st.selectbox("Weather", ["clear","rainy","foggy","hot","cold","stormy"])
    vehicle = st.selectbox("Vehicle", ["Bike","Car","Truck"])

# ---------------- HAVERSINE ----------------
def haversine(p1, p2):
    R = 6371
    lat1 = math.radians(p1[0])
    lon1 = math.radians(p1[1])
    lat2 = math.radians(p2[0])
    lon2 = math.radians(p2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# ---------------- OPTIMIZATION ----------------
def compute_matrix(points):
    size = len(points)
    matrix = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = int(haversine(points[i], points[j])*1000)
    return matrix

def optimize(points):
    manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    matrix = compute_matrix(points)

    def cb(i, j):
        return matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))

    return route

# ---------------- ROUTE FETCH ----------------
def fetch_route(start, end):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
        res = requests.get(url, timeout=5)
        data = res.json()
        return data["routes"][0]
    except:
        return None

# ---------------- MAP ----------------
with left:
    st.subheader("🗺️ Map")

    delivery_points = [p for p in delivery_points if p != (0.0, 0.0)]
    points = [(pickup_lat, pickup_lon)] + delivery_points

    route_coords = []
    total_distance = 0

    if len(points) > 1:
        order = optimize(points)
        ordered_points = [points[i] for i in order]
    else:
        ordered_points = points

    folium_map = folium.Map(location=[pickup_lat, pickup_lon], zoom_start=11)

    for i in range(len(ordered_points)-1):
        route = fetch_route(ordered_points[i], ordered_points[i+1])

        if route:
            coords = route["geometry"]["coordinates"]
            dist = route["distance"]/1000
            total_distance += dist

            latlon = [[c[1],c[0]] for c in coords]
            route_coords.extend(latlon)
            folium.PolyLine(latlon).add_to(folium_map)
        else:
            dist = haversine(ordered_points[i], ordered_points[i+1])
            total_distance += dist
            folium.PolyLine([ordered_points[i], ordered_points[i+1]], color="red").add_to(folium_map)

    for p in ordered_points:
        folium.Marker(p).add_to(folium_map)

    if delivery_points:
        HeatMap(delivery_points).add_to(folium_map)

    st_folium(folium_map, width=800)

# ---------------- ETA ----------------
traffic_factor = {
    "clear": 1,
    "rainy": 1.3,
    "foggy": 1.2,
    "stormy": 1.5,
    "hot": 1.1,
    "cold": 1.1
}

# 🚗 Vehicle speed ranges (urban conditions)
vehicle_speed_range = {
    "Bike": (15, 25),
    "Car": (11, 12),
    "Truck": (12, 20)
}

# Choose speed (use average for stability)
speed = sum(vehicle_speed_range[vehicle]) / 2

# ⏱️ ETA calculation
eta = (total_distance / speed) * 60 * traffic_factor[weather]

# ---------------- DASHBOARD ----------------
c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div class="card">
    <div style="font-size:13px; color:#6b7280;">Distance</div>
    <div style="font-size:26px; font-weight:600;">{round(total_distance,2)} km</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="card">
    <div style="font-size:13px; color:#6b7280;">ETA</div>
    <div style="font-size:26px; font-weight:600;">{int(eta)} min</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="card">
    <div style="font-size:13px; color:#6b7280;">Stops</div>
    <div style="font-size:26px; font-weight:600;">{len(delivery_points)}</div>
</div>
""", unsafe_allow_html=True)
# ---------------- TRACKING ----------------
# ---------------- SMOOTH TRACKING FIX ----------------
import pydeck as pdk

map_placeholder = st.empty()   
status = st.empty()
progress = st.progress(0)

if st.button("▶️ Start Live Tracking") and route_coords:

    route_coords = route_coords[::5]
    path = [[coord[1], coord[0]] for coord in route_coords]

    for i in range(len(path)):
        current_path = path[:i+1]
        current_point = path[i]

        # Progress
        p = i / len(path)
        progress.progress(min(p, 1.0))
        status.info(f"🚚 Progress {int(p*100)}%")

        # Route
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": current_path}],
            get_path="path",
            width_scale=10,
            width_min_pixels=3,
            get_color=[0, 0, 255],
        )

        # Truck
        truck_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[{"position": current_point}],
            get_position="position",
            get_color=[0, 200, 0],
            get_radius=80,
        )

        view_state = pdk.ViewState(
            latitude=current_point[1],
            longitude=current_point[0],
            zoom=13,
        )

        deck = pdk.Deck(
            layers=[path_layer, truck_layer],
            initial_view_state=view_state,
        )

        # ✅ UPDATE SAME MAP (no stacking)
        with map_placeholder:
            st.pydeck_chart(deck)

        time.sleep(0.05)

    progress.progress(1.0)
    status.success("Delivery Completed")




# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
from datetime import datetime

if st.button("🚀 Predict"):

    if total_distance == 0:
        st.warning("Enter valid route")
    else:
        # 🧠 Extra intelligent features (safe for old model)
        now = datetime.now()
        hour = now.hour

        is_peak = 1 if (8 <= hour <= 11 or 17 <= hour <= 21) else 0
        is_night = 1 if (hour >= 22 or hour <= 5) else 0

        num_stops = len(delivery_points)

        traffic_score = total_distance * (1.5 if is_peak else 1.0)

        vehicle_map = {"Bike": 1.0, "Car": 0.8, "Truck": 0.6}
        vehicle_efficiency = vehicle_map[vehicle]

        weather_score_map = {
            "clear": 0,
            "hot": 0.2,
            "cold": 0.3,
            "foggy": 0.5,
            "rainy": 0.7,
            "stormy": 1.0
        }
        weather_score = weather_score_map[weather]

        # ✅ Build dataframe (SAFE with your model)
        df = pd.DataFrame([{
            "latitude": pickup_lat,
            "longitude": pickup_lon,
            "distance_km": total_distance,
            "num_stops": num_stops,
            "hour": hour,
            "is_peak": is_peak,
            "is_night": is_night,
            "traffic_score": traffic_score,
            "vehicle_efficiency": vehicle_efficiency,
            "weather_score": weather_score,
            f"weather_condition_{weather}": 1
        }]).reindex(columns=columns, fill_value=0)

        pred = model.predict(df)
        st.session_state.prediction = int(pred[0])

        # ✅ Confidence (if model supports it)
        try:
            proba = model.predict_proba(df)[0][1]
            st.session_state.confidence = proba
        except:
            st.session_state.confidence = None


# ---------------- RESULT DISPLAY ----------------
if st.session_state.prediction is not None:

    if st.session_state.prediction == 1:
        if st.session_state.get("confidence") is not None:
            st.success(f"🟢 On Time ({round(st.session_state.confidence*100)}% confidence)")
        else:
            st.success("🟢 On Time")

    else:
        if st.session_state.get("confidence") is not None:
            st.error(f"🔴 Delayed ({round((1-st.session_state.confidence)*100)}% confidence)")
        else:
            st.error("🔴 Delayed")
# ---------------- INSIGHTS ----------------
# ---------------- 📈 INTERACTIVE INSIGHTS (NO PIP - ALTAIR) ----------------
import altair as alt

st.subheader("📈 Delivery Insights ")

# 🎯 Calculations
avg_speed = round(total_distance / (eta/60 + 0.01), 2)
delay_risk = min(int((eta / (total_distance + 0.1)) * 10), 100)

weather_impact = {
    "clear": 10, "hot": 20, "cold": 25,
    "foggy": 40, "rainy": 60, "stormy": 80
}
vehicle_efficiency_map = {"Bike": 90, "Car": 75, "Truck": 60}

weather_score = weather_impact[weather]
vehicle_score = vehicle_efficiency_map[vehicle]

# ---------------- 🎨 KPI CARDS ----------------
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class="card">
    <div style="font-size:13px; color:#6b7280;">Avg Speed</div>
    <div style="font-size:26px; font-weight:600;">{avg_speed} km/h</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="card">
    <div style="font-size:13px; color:#6b7280;">Delay Risk</div>
    <div style="font-size:26px; font-weight:600;">{delay_risk}%</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="card">
    <div style="font-size:13px; color:#6b7280;">Weather Impact</div>
    <div style="font-size:26px; font-weight:600;">{weather_score}%</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="card">
    <div style="font-size:13px; color:#6b7280;">Vehicle Efficiency</div>
    <div style="font-size:26px; font-weight:600;">{vehicle_score}%</div>
</div>
""", unsafe_allow_html=True)
# ---------------- 📊 INTERACTIVE LINE CHART ----------------
st.markdown("### 📉 Delivery Performance Trend")

x = list(range(1, 21))
y = []

for i in x:
    value = int((i * 5) - (weather_score * 0.3) - (len(delivery_points) * 2))
    value = max(10, min(value, 100))
    y.append(value)

df_line = pd.DataFrame({
    "Step": x,
    "Performance": y
})

line_chart = alt.Chart(df_line).mark_line(point=True).encode(
    x=alt.X("Step", title="Progress Step"),
    y=alt.Y("Performance", title="Performance %"),
    tooltip=["Step", "Performance"]
).interactive()  # 🔥 enables zoom + pan

st.altair_chart(line_chart, use_container_width=True)

# ---------------- 📊 INTERACTIVE BAR ----------------
st.markdown("### 📊 Delay Factors Breakdown")

factors = ["Traffic", "Weather", "Distance", "Stops"]
values = [
    int(delay_risk * 0.4),
    weather_score,
    int(total_distance * 5),
    len(delivery_points) * 10
]

df_bar = pd.DataFrame({
    "Factor": factors,
    "Impact": values
})

bar_chart = alt.Chart(df_bar).mark_bar().encode(
    x=alt.X("Factor", sort=None),
    y=alt.Y("Impact"),
    tooltip=["Factor", "Impact"],
    color="Factor"
).interactive()

st.altair_chart(bar_chart, use_container_width=True)

# ---------------- 🧠 AI INSIGHTS ----------------
st.markdown("### 🧠  Insights")

if delay_risk > 70:
    st.error("⚠️ High risk of delay due to traffic and route complexity")
elif delay_risk > 40:
    st.warning("⚠️ Moderate delay expected — consider optimizing route")
else:
    st.success("✅ Delivery likely to be on time")

if weather in ["rainy", "stormy"]:
    st.info("🌧 Weather significantly impacting delivery")

if vehicle == "Truck" and total_distance < 5:
    st.info("🚛 Truck is not optimal for short routes")

if len(delivery_points) > 3:
    st.info("📦 Multiple stops increasing delivery time")# ---------------- SUGGESTIONS ----------------
st.subheader("🤖 Suggestions")

if weather in ["rainy","stormy"]:
    st.warning("Bad weather may delay delivery")

if total_distance > 10:
    st.warning("Long route detected")

if vehicle == "Bike" and total_distance > 8:
    st.info("Consider Car instead of Bike")
