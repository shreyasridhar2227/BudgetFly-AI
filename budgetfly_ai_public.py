"""
BudgetFly AI — Cheap Flight Ticket Analyzer

Run:
python budgetfly_ai_clean_final.py --query "Find the cheapest flights from NYC to Atlanta next week"

Optional:
python budgetfly_ai_clean_final.py --query "Find the cheapest flights from NYC to Atlanta next week" --max-results 20
python budgetfly_ai_clean_final.py --query "Find the cheapest flights from NYC to Atlanta next week" --days 7
python budgetfly_ai_clean_final.py --query "Find the cheapest flights from NYC to Atlanta next week" --debug-live
"""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI


SERPAPI_URL = "https://serpapi.com/search.json"
OUTPUT_DIR = Path("output")
SHORT_HISTORY_FILE = Path("flights.csv")
OPENAI_MODEL = "gpt-5-nano"

AIRPORT_GROUPS = {
    "nyc": "JFK,LGA,EWR", "new york": "JFK,LGA,EWR", "new york city": "JFK,LGA,EWR",
    "jfk": "JFK", "lga": "LGA", "ewr": "EWR", "atlanta": "ATL", "atl": "ATL",
    "chicago": "ORD,MDW", "chi": "ORD,MDW", "ord": "ORD", "mdw": "MDW",
    "los angeles": "LAX", "la": "LAX", "lax": "LAX", "san francisco": "SFO", "sf": "SFO", "sfo": "SFO",
    "boston": "BOS", "bos": "BOS", "miami": "MIA", "mia": "MIA", "seattle": "SEA", "sea": "SEA",
    "dallas": "DFW,DAL", "dfw": "DFW", "dal": "DAL", "las vegas": "LAS", "vegas": "LAS",
    "orlando": "MCO", "washington dc": "DCA,IAD,BWI", "washington": "DCA,IAD,BWI", "dc": "DCA,IAD,BWI",
}

BOOKING_BINS = [
    (-1, 3, "0-3 days before departure"), (3, 7, "4-7 days before departure"),
    (7, 14, "8-14 days before departure"), (14, 21, "15-21 days before departure"),
    (21, 35, "22-35 days before departure"), (35, 60, "36-60 days before departure"),
    (60, 10_000, "60+ days before departure"),
]

DEPARTURE_BUCKETS = [
    (0, 5, "Late night"), (5, 8, "Early morning"), (8, 12, "Morning"),
    (12, 17, "Afternoon"), (17, 21, "Evening"), (21, 24, "Late evening"),
]

TRAVEL_NOTES = [
    "Live fares should be trusted more than historical averages because airline prices change quickly.",
    "Historical prices are useful for booking guidance, but they do not guarantee the future price of a route.",
    "For metro areas like New York City, checking nearby airports such as JFK, LGA, and EWR can reveal cheaper fares.",
    "The cheapest option should be selected from valid numeric live prices returned by the API.",
]


@dataclass
class LiveOffer:
    travel_date: str
    weekday: str
    airline: str
    departure_airport: str
    arrival_airport: str
    departure_time: str
    arrival_time: str
    duration_minutes: Optional[int]
    stops: Optional[int]
    price: float
    currency: str
    source_query: str
    destination_query: str
    source: str


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def duration_text(minutes: Optional[int]) -> str:
    if minutes is None or pd.isna(minutes):
        return "N/A"
    minutes = int(minutes)
    return f"{minutes // 60}h {minutes % 60}m"


def parse_price(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def airport_list(group: str) -> List[str]:
    return [x.strip().upper() for x in str(group).split(",") if x.strip()]


def normalize_airports(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = str(text).strip().lower()
    return AIRPORT_GROUPS.get(cleaned, str(text).strip().upper())


def next_monday(today: Optional[date] = None) -> date:
    today = today or date.today()
    days_until = (7 - today.weekday()) % 7
    return today + timedelta(days=days_until or 7)


def table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return ""
    all_rows = [headers] + rows
    widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(headers))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line]
    out.append("|" + "|".join(f" {headers[i]:<{widths[i]}} " for i in range(len(headers))) + "|")
    out.append(line)
    for row in rows:
        out.append("|" + "|".join(f" {str(row[i]):<{widths[i]}} " for i in range(len(headers))) + "|")
    out.append(line)
    return "\n".join(out)


def fallback_parse_request(query: str, days_override: Optional[int] = None) -> Dict[str, Any]:
    q = query.lower()
    source = destination = None

    match = re.search(
        r"from\s+(.+?)\s+to\s+(.+?)(?:\s+next|\s+this|\s+on|\s+for|\s+under|\s+within|\s+and|\s*$)", q
    )
    if match:
        source = normalize_airports(match.group(1))
        destination = normalize_airports(match.group(2))

    if not source or not destination:
        found = []
        for alias, airport_group in AIRPORT_GROUPS.items():
            if re.search(rf"\b{re.escape(alias)}\b", q) and airport_group not in found:
                found.append(airport_group)
        if len(found) >= 2:
            source, destination = found[0], found[1]

    source = source or "JFK,LGA,EWR"
    destination = destination or "ORD,MDW"

    if "next month" in q or "month" in q:
        start, days = date.today() + timedelta(days=21), 21
    elif "next week" in q or "week" in q:
        start, days = next_monday(), 10
    else:
        start, days = date.today() + timedelta(days=14), 14

    exact = re.search(r"(20\d{2}-\d{2}-\d{2})", query)
    if exact:
        start, days = datetime.strptime(exact.group(1), "%Y-%m-%d").date(), 7

    if days_override:
        days = days_override

    return {"source": source, "destination": destination, "start_date": start.isoformat(),
            "days_to_check": days, "currency": "USD", "parse_method": "python_fallback"}


def parse_request_with_llm(query: str, days_override: Optional[int] = None) -> Dict[str, Any]:
    """LLM task: convert natural-language request into structured JSON."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_parse_request(query, days_override)

    today = date.today().isoformat()
    developer_prompt = f"""
You convert a flight-search request into JSON only. Today is {today}.
Use these airport mappings when relevant: {json.dumps(AIRPORT_GROUPS)}.
Return exactly this JSON schema:
{{"source":"airport codes","destination":"airport codes","start_date":"YYYY-MM-DD","days_to_check":number,"currency":"USD"}}
Rules:
- For NYC/New York, use JFK,LGA,EWR unless the user names one specific airport.
- For next week, start on the next Monday and use 10 days.
- For next month, start 21 days from today and use 21 days.
- If the user gives an exact YYYY-MM-DD date, use that date and 7 days unless they specify otherwise.
- Return JSON only. No markdown. No commentary.
""".strip()

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "low"},
            input=[{"role": "developer", "content": developer_prompt}, {"role": "user", "content": query}],
        )
        data = json.loads(response.output_text.strip())
        source = normalize_airports(data.get("source")) or "JFK,LGA,EWR"
        destination = normalize_airports(data.get("destination")) or "ORD,MDW"
        start_date = data.get("start_date")
        datetime.strptime(start_date, "%Y-%m-%d")
        return {"source": source, "destination": destination, "start_date": start_date,
                "days_to_check": int(days_override or data.get("days_to_check") or 10),
                "currency": data.get("currency", "USD"), "parse_method": "llm_json"}
    except Exception:
        return fallback_parse_request(query, days_override)


def extract_offers(data: Dict[str, Any], meta: Dict[str, str]) -> List[LiveOffer]:
    flight_rows = []
    for key in ["best_flights", "other_flights"]:
        if isinstance(data.get(key), list):
            flight_rows.extend(data[key])

    offers: List[LiveOffer] = []
    for row in flight_rows:
        price = parse_price(row.get("price"))
        flights = row.get("flights") or []
        if price is None or price <= 0 or not flights:
            continue
        first, last = flights[0], flights[-1]
        dep, arr = first.get("departure_airport") or {}, last.get("arrival_airport") or {}
        offers.append(
            LiveOffer(
                travel_date=meta["outbound_date"],
                weekday=datetime.strptime(meta["outbound_date"], "%Y-%m-%d").strftime("%A"),
                airline=first.get("airline") or row.get("airline") or "Unknown airline",
                departure_airport=dep.get("id") or dep.get("name") or meta["departure_id"],
                arrival_airport=arr.get("id") or arr.get("name") or meta["arrival_id"],
                departure_time=dep.get("time") or "Not returned",
                arrival_time=arr.get("time") or "Not returned",
                duration_minutes=row.get("total_duration"),
                stops=max(0, len(flights) - 1),
                price=round(float(price), 2),
                currency=meta.get("currency", "USD"),
                source_query=meta["departure_id"],
                destination_query=meta["arrival_id"],
                source="SerpApi Google Flights live result",
            )
        )
    return sorted(offers, key=lambda x: (x.price, x.travel_date, x.departure_time, x.airline))


def serpapi_search(departure_id: str, arrival_id: str, outbound_date: str, currency: str = "USD") -> Tuple[List[LiveOffer], Dict[str, Any]]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is missing from .env")

    params = {
        "engine": "google_flights", "departure_id": departure_id, "arrival_id": arrival_id,
        "outbound_date": outbound_date, "currency": currency, "hl": "en", "gl": "us",
        "type": "2", "travel_class": "1", "adults": "1", "sort_by": "2",
        "show_hidden": "true", "deep_search": "true", "api_key": api_key,
    }
    response = requests.get(SERPAPI_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return extract_offers(data, params), {"url": response.url, "raw_keys": list(data.keys())}


def dedupe_offers(offers: List[LiveOffer]) -> List[LiveOffer]:
    seen, unique = set(), []
    for offer in sorted(offers, key=lambda x: (x.price, x.travel_date, x.departure_time, x.airline)):
        key = (offer.travel_date, offer.price, offer.airline, offer.departure_airport,
               offer.arrival_airport, offer.departure_time, offer.arrival_time, offer.stops)
        if key not in seen:
            seen.add(key)
            unique.append(offer)
    return unique


def search_live_window(source: str, destination: str, start_date: str, days_to_check: int,
                       currency: str, max_results: int) -> Dict[str, Any]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    all_offers: List[LiveOffer] = []
    attempts: List[Dict[str, Any]] = []

    for offset in range(days_to_check):
        travel_date = (start + timedelta(days=offset)).isoformat()
        try:
            offers, meta = serpapi_search(source, destination, travel_date, currency)
            all_offers.extend(offers)
            attempts.append({"date": travel_date, "offers_found": len(offers),
                             "cheapest_price": offers[0].price if offers else None,
                             "source_url": meta.get("url")})
        except Exception as exc:
            attempts.append({"date": travel_date, "offers_found": 0, "error": str(exc)})

    all_offers = dedupe_offers(all_offers)
    prices = [offer.price for offer in all_offers]
    median_price = round(float(median(prices)), 2) if prices else None

    if prices:
        min_price = min(prices)
        cheap_cutoff = min_price * 1.25
        cheap_offers = [offer for offer in all_offers if offer.price <= cheap_cutoff]
        if len(cheap_offers) < min(max_results, len(all_offers)):
            cheap_offers = all_offers[:max_results]
        else:
            cheap_offers = cheap_offers[:max_results]
    else:
        cheap_offers = []

    best = cheap_offers[0] if cheap_offers else None
    return {
        "route": f"{source} → {destination}",
        "window": f"{start_date} for {days_to_check} day(s)",
        "offers_found": len(all_offers),
        "median_live_price_in_window": median_price,
        "cheapest_price": round(best.price, 2) if best else None,
        "savings_vs_median": round(median_price - best.price, 2) if best and median_price is not None else None,
        "best_cheapest_flight": asdict(best) if best else None,
        "cheap_flights": [asdict(offer) for offer in cheap_offers],
        "all_live_offers": [asdict(offer) for offer in all_offers],
        "attempts": attempts,
    }


def booking_window(days_before: float) -> str:
    for low, high, label in BOOKING_BINS:
        if low < days_before <= high:
            return label
    return "Unknown"


def departure_time_bucket(value: Any) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return "Unknown"
    hour = int(dt.hour)
    for low, high, label in DEPARTURE_BUCKETS:
        if low <= hour < high:
            return label
    return "Unknown"


def group_price_summary(df: pd.DataFrame, column: str, price_col: str = "price", limit: int = 7) -> List[Dict[str, Any]]:
    if df.empty or column not in df.columns:
        return []
    grouped = (df.groupby(column)[price_col]
               .agg(records="count", median_fare="median", average_fare="mean")
               .reset_index()
               .sort_values(["median_fare", "records"], ascending=[True, False]))
    return [{column: str(row[column]), "records": int(row["records"]),
             "median_fare": round(float(row["median_fare"]), 2),
             "average_fare": round(float(row["average_fare"]), 2)}
            for _, row in grouped.head(limit).iterrows()]


def analyze_flights_csv(path: Path, source_airports: List[str], dest_airports: List[str]) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False, "message": f"Historical file not found: {path.name}"}

    usecols = ["Searched Date", "Departure Date", "Departure Airport", "Arrival Airport", "Airline", "Price"]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    required = {"Searched Date", "Departure Date", "Departure Airport", "Arrival Airport", "Price"}
    if not required.issubset(df.columns):
        return {"available": False, "message": f"{path.name} missing columns. Found: {list(df.columns)}"}

    df = df.copy()
    df.loc[:, "Departure Airport"] = df["Departure Airport"].astype(str).str.upper()
    df.loc[:, "Arrival Airport"] = df["Arrival Airport"].astype(str).str.upper()

    exact_route = df.loc[df["Departure Airport"].isin(source_airports) & df["Arrival Airport"].isin(dest_airports)].copy()
    if exact_route.empty:
        route_df = df.copy()
        scope_level = "overall_fallback"
        scope_note = "Exact route history was not available, so all flights.csv rows were used for broader guidance."
    else:
        route_df = exact_route.copy()
        scope_level = "exact_route"
        scope_note = "Exact route history from flights.csv was used."

    route_df.loc[:, "searched_date"] = pd.to_datetime(route_df["Searched Date"], errors="coerce")
    route_df.loc[:, "departure_date"] = pd.to_datetime(route_df["Departure Date"], errors="coerce")
    route_df.loc[:, "price"] = route_df["Price"].apply(parse_price)
    route_df = route_df.dropna(subset=["searched_date", "departure_date", "price"]).copy()
    route_df = route_df.loc[route_df["price"] > 0].copy()
    route_df.loc[:, "days_before_departure"] = (route_df["departure_date"].dt.date - route_df["searched_date"].dt.date).apply(lambda delta: delta.days)
    route_df = route_df.loc[(route_df["days_before_departure"] >= 0) & (route_df["days_before_departure"] <= 365)].copy()

    if route_df.empty:
        return {"available": False, "message": "No usable historical rows after cleaning."}

    route_df.loc[:, "booking_window"] = route_df["days_before_departure"].apply(booking_window)
    route_df.loc[:, "search_weekday"] = route_df["searched_date"].dt.day_name()
    route_df.loc[:, "travel_weekday"] = route_df["departure_date"].dt.day_name()
    route_df.loc[:, "departure_bucket"] = route_df["departure_date"].apply(departure_time_bucket)
    route_df.loc[:, "airline"] = route_df["Airline"].astype(str) if "Airline" in route_df.columns else "Unknown"

    typical_median = float(route_df["price"].median())
    booking_windows = group_price_summary(route_df.loc[route_df["booking_window"] != "Unknown"].copy(), "booking_window")
    search_weekdays = group_price_summary(route_df.copy(), "search_weekday")
    travel_weekdays = group_price_summary(route_df.copy(), "travel_weekday")
    departure_buckets = group_price_summary(route_df.loc[route_df["departure_bucket"] != "Unknown"].copy(), "departure_bucket")
    airlines = group_price_summary(route_df.loc[route_df["airline"] != "Unknown"].copy(), "airline")
    best_window = booking_windows[0] if booking_windows else None
    estimated_savings = typical_median - best_window["median_fare"] if best_window else None

    return {
        "available": True, "records_used": int(len(route_df)), "scope_level": scope_level, "scope_note": scope_note,
        "typical_median_fare": round(typical_median, 2), "best_booking_window": best_window,
        "best_search_day": search_weekdays[0] if search_weekdays else None,
        "best_travel_day": travel_weekdays[0] if travel_weekdays else None,
        "best_departure_bucket": departure_buckets[0] if departure_buckets else None,
        "best_airline": airlines[0] if airlines else None,
        "estimated_savings_vs_typical": round(float(estimated_savings), 2) if estimated_savings is not None else None,
        "booking_window_table": booking_windows, "search_weekday_table": search_weekdays,
        "travel_weekday_table": travel_weekdays, "departure_bucket_table": departure_buckets, "airline_table": airlines,
    }


def generate_llm_recommendation(query: str, live: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    grounding = {
        "user_query": query,
        "best_cheapest_flight": live.get("best_cheapest_flight"),
        "cheap_flights": live.get("cheap_flights", [])[:10],
        "median_live_price": live.get("median_live_price_in_window"),
        "savings_vs_median": live.get("savings_vs_median"),
        "historical_guidance_from_flights_csv": history,
        "travel_notes": TRAVEL_NOTES,
    }
    if not api_key:
        return {"available": False, "message": "OPENAI_API_KEY missing. Skipped LLM recommendation.", "grounding_data": grounding}

    developer_prompt = """
You are BudgetFly AI's LLM insight generator.
Use ONLY the supplied JSON. Do not invent prices, airlines, dates, airports, or savings.
Do not assign scores. Do not rank using hidden criteria.
Explain:
1. the best cheapest live flight
2. why it was selected
3. what other cheap live options are available
4. what flights.csv suggests historically about booking/travel timing
5. one caution that live fares change quickly
Keep the answer under 220 words.
""".strip()
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL, reasoning={"effort": "low"},
            input=[{"role": "developer", "content": developer_prompt}, {"role": "user", "content": json.dumps(grounding, indent=2, default=str)}],
        )
        return {"available": True, "model": OPENAI_MODEL, "text": response.output_text.strip(), "grounding_data": grounding}
    except Exception as exc:
        return {"available": False, "message": str(exc), "grounding_data": grounding}


def cheap_flights_rows(offers: List[Dict[str, Any]]) -> List[List[str]]:
    rows = []
    for i, offer in enumerate(offers, start=1):
        rows.append([str(i), f"{offer['travel_date']} ({offer['weekday'][:3]})", offer["airline"],
                     f"{offer['departure_airport']} → {offer['arrival_airport']}", offer["departure_time"],
                     offer["arrival_time"], duration_text(offer.get("duration_minutes")), str(offer.get("stops")), money(offer["price"])])
    return rows


def build_report(query: str, request_data: Dict[str, Any], live: Dict[str, Any], history: Dict[str, Any], llm: Dict[str, Any], debug_live: bool) -> str:
    best, cheap = live.get("best_cheapest_flight"), live.get("cheap_flights", [])
    lines: List[str] = []
    lines.append("╔════════════════════════════════════════════════════════════╗")
    lines.append("║              BudgetFly AI — Cheap Flight Analyzer         ║")
    lines.append("╚════════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f"Request       : {query}")
    lines.append(f"Route checked : {request_data['source']} → {request_data['destination']}")
    lines.append(f"Date window   : {request_data['start_date']} for {request_data['days_to_check']} days")
    lines.append(f"Parser used   : {request_data.get('parse_method')}")
    lines.append("Data used     : SerpApi live tracking + flights.csv historical data")
    lines.append("")
    lines.append("BEST CHEAPEST LIVE FLIGHT")
    lines.append("-------------------------")
    if best:
        rows = [["Date", f"{best['travel_date']} ({best['weekday']})"], ["Airline", best["airline"]],
                ["Route", f"{best['departure_airport']} → {best['arrival_airport']}"], ["Departure", best["departure_time"]],
                ["Arrival", best["arrival_time"]], ["Duration", duration_text(best.get("duration_minutes"))],
                ["Stops", str(best.get("stops"))], ["Price", f"{money(best['price'])} {best.get('currency', 'USD')}"]]
        lines.append(table(rows, ["Field", "Value"]))
    else:
        lines.append("No valid live fares were found for this route/date window.")
    lines.append("")
    lines.append("ALL AVAILABLE CHEAP LIVE FLIGHTS")
    lines.append("--------------------------------")
    if cheap:
        lines.append(table(cheap_flights_rows(cheap), ["#", "Date", "Airline", "Route", "Depart", "Arrive", "Duration", "Stops", "Price"]))
    else:
        lines.append("No cheap live flights available.")
    lines.append("")
    lines.append("LIVE PRICE SUMMARY")
    lines.append("------------------")
    lines.append(table([["Total valid live offers found", str(live.get("offers_found", 0))],
                        ["Median live fare in window", money(live.get("median_live_price_in_window"))],
                        ["Cheapest live fare", money(live.get("cheapest_price"))],
                        ["Savings vs median live fare", money(live.get("savings_vs_median"))]], ["Metric", "Value"]))
    lines.append("")
    lines.append("HISTORICAL GUIDANCE FROM flights.csv")
    lines.append("------------------------------------")
    if history.get("available"):
        hist_rows = []
        mapping = [("Best booking window", "best_booking_window", "booking_window"), ("Best search/book day", "best_search_day", "search_weekday"),
                   ("Best travel day", "best_travel_day", "travel_weekday"), ("Best departure time", "best_departure_bucket", "departure_bucket"),
                   ("Historically cheaper airline", "best_airline", "airline")]
        for label, key, value_col in mapping:
            row = history.get(key)
            if row:
                hist_rows.append([label, str(row[value_col]), money(row["median_fare"]), str(row["records"])])
        lines.append(table(hist_rows, ["Insight", "Best option", "Median fare", "Rows"]))
        lines.append(f"Historical records used : {history.get('records_used')}")
        lines.append(f"Typical historical fare : {money(history.get('typical_median_fare'))}")
        lines.append(f"Scope note              : {history.get('scope_note')}")
    else:
        lines.append(history.get("message", "Historical data unavailable."))
    lines.append("")
    lines.append("LLM-BASED EXPLANATION")
    lines.append("---------------------")
    lines.append(llm.get("text", f"LLM explanation unavailable: {llm.get('message')}"))
    if debug_live:
        lines.append("")
        lines.append("LIVE SEARCH DEBUG")
        lines.append("-----------------")
        for attempt in live.get("attempts", []):
            lines.append(f"- {attempt}")
    return "\n".join(lines)


def evaluate_accuracy(live: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
    all_offers, cheap, best = live.get("all_live_offers", []), live.get("cheap_flights", []), live.get("best_cheapest_flight")
    valid_prices = [parse_price(offer.get("price")) for offer in all_offers]
    valid_prices = [p for p in valid_prices if p is not None and p > 0]
    best_is_lowest = bool(best and valid_prices and float(best["price"]) == min(valid_prices))
    return {
        "live_price_accuracy": {
            "total_live_offers_saved": len(all_offers), "valid_numeric_prices": len(valid_prices),
            "price_extraction_rate": round(len(valid_prices) / max(1, len(all_offers)), 3),
            "cheapest_live_price": round(min(valid_prices), 2) if valid_prices else None,
            "best_selected_price": round(float(best["price"]), 2) if best else None,
            "best_selected_is_lowest_live_fare": best_is_lowest, "cheap_flights_returned": len(cheap),
        },
        "historical_guidance_accuracy": {
            "flights_csv_available": bool(history.get("available")), "records_used": history.get("records_used"),
            "scope_level": history.get("scope_level"), "typical_median_fare": history.get("typical_median_fare"),
            "best_booking_window": history.get("best_booking_window"), "best_search_day": history.get("best_search_day"),
            "best_travel_day": history.get("best_travel_day"), "best_departure_bucket": history.get("best_departure_bucket"),
            "best_airline": history.get("best_airline"),
        },
        "method_note": "Live fare accuracy is checked from SerpApi JSON. Historical guidance is computed from flights.csv. The LLM is used for query parsing and written explanation only, not numeric scoring.",
    }


def save_outputs(report: str, request_data: Dict[str, Any], live: Dict[str, Any], history: Dict[str, Any], llm: Dict[str, Any], accuracy: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = {
        "budgetfly_final_answer.txt": report,
        "budgetfly_request.json": json.dumps(request_data, indent=2, default=str),
        "budgetfly_all_live_offers.json": json.dumps(live.get("all_live_offers", []), indent=2, default=str),
        "budgetfly_cheap_flights.json": json.dumps(live.get("cheap_flights", []), indent=2, default=str),
        "budgetfly_best_cheapest_flight.json": json.dumps(live.get("best_cheapest_flight", {}), indent=2, default=str),
        "budgetfly_live_attempts.json": json.dumps(live.get("attempts", []), indent=2, default=str),
        "flights_csv_historical_analysis.json": json.dumps(history, indent=2, default=str),
        "llm_recommendation.json": json.dumps(llm, indent=2, default=str),
        "output_accuracy_analysis.json": json.dumps(accuracy, indent=2, default=str),
    }
    for filename, content in files.items():
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")
    acc = accuracy["live_price_accuracy"]
    hist = accuracy["historical_guidance_accuracy"]
    text = "\n".join([
        "BudgetFly AI — Output Accuracy Analysis", "=======================================", "",
        "LIVE PRICE ACCURACY", "-------------------",
        f"Total live offers saved      : {acc['total_live_offers_saved']}",
        f"Valid numeric prices         : {acc['valid_numeric_prices']}",
        f"Price extraction rate        : {acc['price_extraction_rate']}",
        f"Cheapest live price          : {money(acc['cheapest_live_price'])}",
        f"Best selected price          : {money(acc['best_selected_price'])}",
        f"Best selected is lowest fare : {acc['best_selected_is_lowest_live_fare']}",
        f"Cheap flights returned       : {acc['cheap_flights_returned']}", "",
        "HISTORICAL GUIDANCE ACCURACY", "----------------------------",
        f"flights.csv available        : {hist['flights_csv_available']}",
        f"Records used                 : {hist['records_used']}",
        f"Scope level                  : {hist['scope_level']}",
        f"Typical median fare          : {money(hist['typical_median_fare'])}", "",
        "METHOD NOTE", "-----------", accuracy["method_note"],
    ])
    (OUTPUT_DIR / "output_accuracy_analysis.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="BudgetFly AI: SerpApi live tracking + flights.csv history + LLM explanation")
    parser.add_argument("--query", required=True, help="Natural-language flight request")
    parser.add_argument("--days", type=int, default=None, help="Override number of days to check")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--short-history", default=str(SHORT_HISTORY_FILE))
    parser.add_argument("--max-results", type=int, default=15, help="Maximum cheap flights to display")
    parser.add_argument("--debug-live", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=True)
    request_data = parse_request_with_llm(args.query, args.days)
    request_data["source_airports"] = airport_list(request_data["source"])
    request_data["destination_airports"] = airport_list(request_data["destination"])
    request_data["currency"] = args.currency

    live = search_live_window(request_data["source"], request_data["destination"], request_data["start_date"],
                              request_data["days_to_check"], args.currency, args.max_results)
    history = analyze_flights_csv(Path(args.short_history), request_data["source_airports"], request_data["destination_airports"])
    llm = generate_llm_recommendation(args.query, live, history)
    accuracy = evaluate_accuracy(live, history)
    report = build_report(args.query, request_data, live, history, llm, args.debug_live)
    print(report)
    save_outputs(report, request_data, live, history, llm, accuracy)


if __name__ == "__main__":
    main()
