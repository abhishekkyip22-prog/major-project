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

# ---------------- LOAD MODEL ----------------
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(base_path, "columns.pkl"), "rb"))

# ---------------- SESSION ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ---------------- PAGE ----------------
st.set_page_config(page_title="Smart Delivery System", layout="wide")
st.title("🚚 Smart Delivery Routing System")

left, right = st.columns([3, 1])

# ---------------- INPUT ----------------
with right:
    st.subheader("📦 Setup")

    pickup_lat = st.number_input("Pickup Latitude", value=28.6)
    pickup_lon = st.number_input("Pickup Longitude", value=77.2)

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
traffic_factor = {"clear":1,"rainy":1.3,"foggy":1.2,"stormy":1.5,"hot":1.1,"cold":1.1}
eta = total_distance * 3 * traffic_factor[weather]

# ---------------- DASHBOARD ----------------
c1,c2,c3 = st.columns(3)
c1.metric("Distance (km)", round(total_distance,2))
c2.metric("ETA (min)", int(eta))
c3.metric("Stops", len(delivery_points))

# ---------------- TRACKING ----------------
placeholder = st.empty()
status = st.empty()
progress = st.progress(0)

if st.button("▶️ Start Live Tracking") and route_coords:
    route_coords = route_coords[::5]

    for i, coord in enumerate(route_coords):
        p = i/len(route_coords)
        progress.progress(min(p,1.0))

        sim_map = folium.Map(location=coord, zoom_start=13)
        folium.Marker(coord, icon=folium.Icon(color="red")).add_to(sim_map)

        with placeholder:
            st_folium(sim_map, width=800, key=f"track{i}")

        status.info(f"🚚 Progress {int(p*100)}%")
        time.sleep(0.03)

    progress.progress(1.0)
    status.success("Delivery Completed")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict"):

    if total_distance == 0:
        st.warning("Enter valid route")
    else:
        df = pd.DataFrame([{
            "latitude": pickup_lat,
            "longitude": pickup_lon,
            "distance_km": total_distance,
            f"weather_condition_{weather}": 1
        }]).reindex(columns=columns, fill_value=0)

        pred = model.predict(df)
        st.session_state.prediction = int(pred[0])

if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.success("🟢 On Time")
    else:
        st.error("🔴 Delayed")

# ---------------- INSIGHTS ----------------
st.subheader("📈 Insights")

st.line_chart(np.random.randint(60,95,7))

fig, ax = plt.subplots()
ax.bar(["Traffic","Weather","Distance"], [40,30,30])
st.pyplot(fig)

# ---------------- SUGGESTIONS ----------------
st.subheader("🤖 Suggestions")

if weather in ["rainy","stormy"]:
    st.warning("Bad weather may delay delivery")

if total_distance > 10:
    st.warning("Long route detected")

if vehicle == "Bike" and total_distance > 8:
    st.info("Consider Car instead of Bike")