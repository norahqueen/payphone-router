from __future__ import annotations
import json
import multiprocessing
import os
import time
import osrm
from dotenv import load_dotenv
from typing import TypedDict

import requests
from geopy.distance import geodesic
from ortools.sat.python import cp_model


class Payphone(TypedDict):
    id: int
    lon: float
    lat: float


def fetch_payphones(username_to_exclude: str | None) -> list[Payphone]:
    PAYPHONE_API_URL = "https://payphonetag.com/api/payphones"
    response = requests.get(PAYPHONE_API_URL)
    response.raise_for_status()
    data = response.json()
    ids_to_exclude: set[str] = set()
    if username_to_exclude:
        cell_id = None
        for player_id, player_data in data["players"].items():
            if player_data.get("name") == username_to_exclude:
                ids_to_exclude.add(player_id)
                cell_id = player_data.get("cellId")
                print(
                    f"Excluding payphones held by player '{username_to_exclude}' (id {player_id})"
                )
                break
        if cell_id is not None:
            for player_id, player_data in data["players"].items():
                if (
                    player_data.get("cellId") == cell_id
                    and player_id not in ids_to_exclude
                ):
                    ids_to_exclude.add(player_id)
            print(
                f"Also excluding payphones held by cell {cell_id} members ({len(ids_to_exclude) - 1} cell-mates)"
            )
    phones: list[Payphone] = []
    for p in data["payphones"]:
        if p[4] != "active":
            # p[4] is status - we only want active payphones.
            continue
        if ids_to_exclude and str(p[3]) in ids_to_exclude:
            # p[3] is holder_id - skip if held by the user or a cell-mate.
            continue
        phones.append({"id": p[0], "lon": p[1], "lat": p[2]})
    return phones


def get_starting_payphone(
    home_coordinates: tuple[float, float], payphones: list[Payphone]
) -> Payphone:
    home_lat, home_lon = home_coordinates
    nearest = None
    nearest_distance = float("inf")
    for p in payphones:
        dist = geodesic((home_lat, home_lon), (p["lat"], p["lon"])).meters
        if dist < nearest_distance:
            nearest = p
            nearest_distance = dist
    if nearest is None:
        raise ValueError("No payphones found!")
    return nearest


def filter_payphones(
    centre: tuple[float, float],
    payphones: list[Payphone],
    radius_m: float,
    max_latitude: float | None = None,
) -> list[Payphone]:
    filtered = []
    for p in payphones:
        dist = geodesic(centre, (p["lat"], p["lon"])).meters
        if dist <= radius_m and (max_latitude is None or p["lat"] <= max_latitude):
            filtered.append(p)
    return filtered


def get_distance_matrix(payphones: list[Payphone]) -> list[list[float]]:
    OSRM_PATH = "./routing/nsw_osm"
    engine = osrm.OSRM(OSRM_PATH)
    params = osrm.TableParameters(
        coordinates=[
            (p["lon"], p["lat"]) for p in payphones
        ],  # list of coords we want in our matrix
        annotations=[
            "distance"
        ],  # The api returns duration as well, but it's not relevant since we'll be running.
    )
    result = engine.Table(params)
    return result["distances"]


def _greedy_nearest_neighbour(
    start: int,
    num_nodes: int,
    scaled_dist: list[list[int]],
    scaled_budget: int,
    UNREACHABLE: int,
) -> tuple[list[int], set[int]]:
    """Greedy nearest-neighbour route used as a solution hint for the CP-SAT solver."""
    visited: set[int] = {start}
    route = [start]
    remaining = scaled_budget
    current = start
    while True:
        best_j, best_d = None, UNREACHABLE
        for j in range(num_nodes):
            d = scaled_dist[current][j]
            if j not in visited and d < UNREACHABLE and d <= remaining and d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            break
        visited.add(best_j)
        route.append(best_j)
        remaining -= best_d
        current = best_j
    return route, visited


def solve_cp_sat(
    start_idx: int,
    all_indices: set[int],
    dist: list[list[float]],
    budget: float,
    time_limit: float = 600.0,
    max_travel_distance_per_leg_metres: float = 1000.0,
) -> list[int] | None:
    """
    Maximise the number of unique payphones visited, subject to a maximum total travel distance.
    Each payphone can be visited at most once. The route does not need to return to the starting location.

    Uses constraint programming (Google OR-Tools CP-SAT solver) to find the best possible route.
    Constraint programming works by declaring variables and rules, then asking the solver to find
    values that satisfy all rules while maximising an objective (here: phones visited).
    """
    # `all_indices` refers to rows/columns in the global distance matrix, which may be a
    # non-contiguous subset. We sort them and map to a compact 0-based local index so that
    # solver variables are numbered 0…N-1. The two dicts let us translate back to original
    # IDs when returning the result.
    sorted_indices = sorted(all_indices)
    num_phones = len(sorted_indices)
    orig_to_local = {orig: local for local, orig in enumerate(sorted_indices)}
    local_to_orig = {local: orig for orig, local in orig_to_local.items()}
    start_local = orig_to_local[start_idx]

    # CP-SAT works exclusively with integers, so we scale all float distances by SCALE to
    # preserve one decimal place of precision. Any leg longer than the per-leg cap is
    # replaced with UNREACHABLE — a sentinel large enough that it can never fit within the
    # budget, effectively removing that edge from the graph.
    SCALE = 10
    UNREACHABLE = 10**9
    scaled_dist = [
        [
            int(round(dist[sorted_indices[i]][sorted_indices[j]] * SCALE))
            if dist[sorted_indices[i]][sorted_indices[j]] < float("inf")
            and dist[sorted_indices[i]][sorted_indices[j]]
            <= max_travel_distance_per_leg_metres
            else UNREACHABLE
            for j in range(num_phones)
        ]
        for i in range(num_phones)
    ]
    scaled_budget = int(round(budget * SCALE))

    # The CpModel object is the container for all decision variables, constraints, and the
    # objective. Everything we declare below is registered on this model.
    model = cp_model.CpModel()

    # For each phone, declare a Boolean decision variable: 1 = included in the route, 0 = skipped.
    # The start phone is fixed to 1 — we always depart from there.
    is_visited = [model.new_bool_var(f"v_{i}") for i in range(num_phones)]
    model.add(is_visited[start_local] == 1)

    # --- Circuit constraint ---
    # `add_circuit` tells CP-SAT that the selected arcs must form a single closed tour
    # covering every node exactly once (a Hamiltonian circuit on the *selected* nodes).
    # We build up the arc list in three stages below.
    DUMMY = num_phones  # index of the fictional "sink" node used to open the tour

    arcs: list[tuple[int, int, cp_model.BoolVarT]] = []

    # Stage 1 — self-loops for unvisited nodes.
    # `add_circuit` requires every node to appear in the tour. For nodes we choose to skip,
    # a self-loop arc (i → i) that is active when is_visited[i] == 0 satisfies this
    # requirement without including the node in the actual route.
    for i in range(num_phones):
        arcs.append((i, i, is_visited[i].negated()))

    # Stage 2 — directed travel arcs between reachable pairs of phones.
    # For every (i, j) where the walk is feasible, we create a Boolean arc variable t_i_j.
    # Setting t_i_j = 1 means the route goes directly from phone i to phone j.
    travel: dict[tuple[int, int], cp_model.BoolVarT] = {}
    for i in range(num_phones):
        for j in range(num_phones):
            if i != j and scaled_dist[i][j] < UNREACHABLE:
                lit = model.new_bool_var(f"t_{i}_{j}")
                travel[(i, j)] = lit
                arcs.append((i, j, lit))

    # Stage 3 — dummy sink node to allow an open-ended route.
    # `add_circuit` demands a *closed* tour, but our route doesn't need to return to start.
    # We solve this by adding a fictional DUMMY node: every real node has an arc leading to
    # it (representing "this is the last stop"), and the DUMMY connects back to start to
    # close the circuit. Exactly one exit_i will be 1 in any valid solution.
    exits_to_sink = [model.new_bool_var(f"exit_{i}") for i in range(num_phones)]
    for i in range(num_phones):
        arcs.append((i, DUMMY, exits_to_sink[i]))

    dummy_to_start = model.new_bool_var("dummy_to_start")
    model.add(
        dummy_to_start == 1
    )  # this arc is always active — it just closes the loop
    arcs.append((DUMMY, start_local, dummy_to_start))

    model.add_circuit(arcs)

    # The sum of distances for all active travel arcs must not exceed the distance budget.
    # This is a standard linear inequality over the arc variables.
    model.add(
        sum(scaled_dist[i][j] * travel[(i, j)] for (i, j) in travel) <= scaled_budget
    )

    # Objective: visit as many phones as possible. CP-SAT will search for the variable
    # assignment that satisfies all constraints and maximises this count.
    model.maximize(sum(is_visited))

    # Warm-start hint: run a fast greedy nearest-neighbour heuristic first to get a
    # reasonable initial solution, then feed it to the solver as a hint. CP-SAT isn't
    # obliged to use it, but starting from a known-good solution lets the solver prune
    # large parts of the search space much earlier, cutting solve time significantly.
    hint_route, hint_visited = _greedy_nearest_neighbour(
        start_local, num_phones, scaled_dist, scaled_budget, UNREACHABLE
    )
    hint_arcs = set(zip(hint_route, hint_route[1:]))
    for i in range(num_phones):
        model.add_hint(is_visited[i], 1 if i in hint_visited else 0)
    for (i, j), lit in travel.items():
        model.add_hint(lit, 1 if (i, j) in hint_arcs else 0)
    last_stop = hint_route[-1]
    for i in range(num_phones):
        model.add_hint(exits_to_sink[i], 1 if i == last_stop else 0)

    # --- Solve ---
    # Run the solver with a wall-clock time limit and as many parallel workers as there are
    # CPU cores — CP-SAT is designed for multi-threaded search.
    # OPTIMAL means the best possible solution was proven; FEASIBLE means a valid solution
    # was found within the time limit but optimality is not guaranteed.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = multiprocessing.cpu_count()
    status = solver.solve(model)
    print(f"Solver finished with status {solver.status_name(status)}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # --- Route reconstruction ---
    # Read which arc variables the solver set to 1 and build a next_node map (i → j).
    # Arcs leading to DUMMY mark the final stop; we use that as a termination sentinel.
    next_node: dict[int, int] = {}
    for (i, j), lit in travel.items():
        if solver.value(lit) == 1:
            next_node[i] = j
    for i, lit in enumerate(exits_to_sink):
        if solver.value(lit) == 1:
            next_node[i] = DUMMY

    # Walk the next_node chain from start to reconstruct the ordered visit sequence.
    ordered_route = [start_local]
    current = start_local
    while True:
        nxt = next_node.get(current)
        if nxt is None or nxt == DUMMY:
            break
        ordered_route.append(nxt)
        current = nxt

    # Translate local 0-based indices back to the original global IDs before returning.
    return [local_to_orig[i] for i in ordered_route]


def call_model(
    payphones: list[Payphone],
    distance_matrix: list[list[float]],
    start_payphone_id: int,
    distance_budget_metres: int,
    max_travel_distance_per_leg_metres: float = 1000.0,
) -> list[int] | None:
    ids = [p["id"] for p in payphones]
    id_to_index = {id_: i for i, id_ in enumerate(ids)}
    route = solve_cp_sat(
        start_idx=id_to_index[start_payphone_id],
        all_indices=set(range(len(ids))),
        dist=distance_matrix,
        budget=distance_budget_metres,
        max_travel_distance_per_leg_metres=max_travel_distance_per_leg_metres,
    )
    return [ids[i] for i in route] if route is not None else None


def get_path_for_route(route_ordered_ids: list[int], payphones: list[Payphone]):
    features = []
    n_segments = len(route_ordered_ids) - 1
    coords = {p["id"]: (p["lon"], p["lat"]) for p in payphones}
    OSRM_PATH = "./routing/nsw_osm"
    engine = osrm.OSRM(OSRM_PATH)

    for i in range(n_segments):
        id_a, id_b = route_ordered_ids[i], route_ordered_ids[i + 1]

        params = osrm.RouteParameters(
            coordinates=[coords[id_a], coords[id_b]],
            geometries="geojson",
            overview="full",
        )
        result = engine.Route(params)
        segment = result["routes"][0]
        geom = segment["geometry"]
        geometry = {
            "type": geom["type"],
            "coordinates": [list(c) for c in geom["coordinates"]],
        }

        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "from_id": id_a,
                    "to_id": id_b,
                    "distance_m": segment["distance"],
                    "duration_s": segment["duration"],
                },
            }
        )

    result_geojson = {"type": "FeatureCollection", "features": features}
    return result_geojson


def write_html(
    payphones: list[Payphone], route_ordered_ids: list[int], path_geojson: dict
):
    route = {"ordered_ids": route_ordered_ids, "path": path_geojson}
    inline_script = (
        "<script>\n"
        f"window.__PAYPHONES__ = {json.dumps(payphones)};\n"
        f"window.__ROUTE__ = {json.dumps(route)};\n"
        "</script>"
    )
    with open("index_template.html") as f:
        html = f.read()
    html = html.replace("<!-- INLINE_DATA -->", inline_script)
    with open("public/index.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    load_dotenv()

    lat, lon = os.environ["HOME_COORDINATES"].split(",")
    HOME_COORDINATES = (float(lat), float(lon))
    PAYPHONE_FILTER_RADIUS_M = 8000
    PLAYER_USERNAME = os.environ["PLAYER_USERNAME"]
    DISTANCE_BUDGET_METRES = 7000
    MAX_LEG_DISTANCE_METRES = 2000
    MAX_LATITUDE: float | None = (
        None  # you can set a latitude cap - I use it to avoid being sent into the city
    )
    START_PAYPHONE_ID_OVERRIDE: int | None = (
        None  # Set to a payphone ID to force a specific start
    )

    payphones = fetch_payphones(PLAYER_USERNAME)
    print(f"Fetched {len(payphones)} active payphones from the Payphone Tag server.")
    payphones = filter_payphones(
        HOME_COORDINATES, payphones, PAYPHONE_FILTER_RADIUS_M, MAX_LATITUDE
    )
    print(
        f"Filtered to {len(payphones)} payphones within {PAYPHONE_FILTER_RADIUS_M} metres of home."
    )
    if START_PAYPHONE_ID_OVERRIDE is not None:
        matches = [p for p in payphones if p["id"] == START_PAYPHONE_ID_OVERRIDE]
        if not matches:
            raise ValueError(
                f"START_PAYPHONE_ID_OVERRIDE={START_PAYPHONE_ID_OVERRIDE} not found in filtered payphones."
            )
        starting_payphone = matches[0]
        print(
            f"Using overridden start payphone {starting_payphone['id']} ({starting_payphone['lat']:.5f}, {starting_payphone['lon']:.5f})"
        )
    else:
        starting_payphone = get_starting_payphone(HOME_COORDINATES, payphones)
        print(
            f"We're starting at payphone {starting_payphone['id']} ({starting_payphone['lat']:.5f}, {starting_payphone['lon']:.5f})"
        )
    t0 = time.time()
    distance_matrix = get_distance_matrix(payphones)
    print(f"Distance matrix calculated in {time.time() - t0:.2f}s.")

    print("Running solver to find optimal route...")
    route = call_model(
        payphones,
        distance_matrix,
        starting_payphone["id"],
        DISTANCE_BUDGET_METRES,
        MAX_LEG_DISTANCE_METRES,
    )

    if route is None:
        print("No route found!")
        raise SystemExit(1)

    path = get_path_for_route(route, payphones)
    total_distance_m = sum(f["properties"]["distance_m"] for f in path["features"])
    print(
        f"Total route distance: {total_distance_m:.0f} m ({total_distance_m / 1000:.2f} km) across {len(route)} payphones."
    )
    write_html(payphones, route, path)
