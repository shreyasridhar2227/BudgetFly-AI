"""
BudgetFly AI — Accuracy Analyzer

Run after the main script:
python budgetfly_ai_public.py --query "Find the cheapest flights from NYC to Atlanta next week"
python accuracy_analyzer.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


OUTPUT_DIR = Path("output")

ALL_LIVE_OFFERS_PATH = OUTPUT_DIR / "budgetfly_all_live_offers.json"
CHEAP_FLIGHTS_PATH = OUTPUT_DIR / "budgetfly_cheap_flights.json"
BEST_FLIGHT_PATH = OUTPUT_DIR / "budgetfly_best_cheapest_flight.json"
HISTORY_ANALYSIS_PATH = OUTPUT_DIR / "flights_csv_historical_analysis.json"
FINAL_ANSWER_PATH = OUTPUT_DIR / "budgetfly_final_answer.txt"

ACCURACY_JSON_PATH = OUTPUT_DIR / "output_accuracy_analysis.json"
ACCURACY_TEXT_PATH = OUTPUT_DIR / "output_accuracy_analysis.txt"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def money_to_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def money(value: Any) -> str:
    number = money_to_float(value)
    if number is None:
        return "N/A"
    return f"${number:,.2f}"


def table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return ""

    all_rows = [headers] + rows
    widths = [
        max(len(str(row[i])) for row in all_rows)
        for i in range(len(headers))
    ]

    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    out = [line]
    out.append(
        "|"
        + "|".join(
            f" {headers[i]:<{widths[i]}} "
            for i in range(len(headers))
        )
        + "|"
    )
    out.append(line)

    for row in rows:
        out.append(
            "|"
            + "|".join(
                f" {str(row[i]):<{widths[i]}} "
                for i in range(len(headers))
            )
            + "|"
        )

    out.append(line)
    return "\n".join(out)


def evaluate_live_accuracy() -> Dict[str, Any]:
    all_offers = load_json(ALL_LIVE_OFFERS_PATH, [])
    cheap_flights = load_json(CHEAP_FLIGHTS_PATH, [])
    best_flight = load_json(BEST_FLIGHT_PATH, {})

    if not all_offers:
        return {
            "available": False,
            "message": "No live offer file found. Run the main project script first.",
            "expected_file": str(ALL_LIVE_OFFERS_PATH),
        }

    valid_offers = []

    for offer in all_offers:
        price = money_to_float(offer.get("price"))
        if price is not None and price > 0:
            copied = dict(offer)
            copied["price"] = price
            valid_offers.append(copied)

    if not valid_offers:
        return {
            "available": False,
            "message": "Live offers were saved, but none had valid numeric prices.",
            "total_live_offers_saved": len(all_offers),
        }

    sorted_valid = sorted(valid_offers, key=lambda x: float(x["price"]))
    cheapest_offer = sorted_valid[0]

    prices = [float(offer["price"]) for offer in sorted_valid]
    median_live_price = float(pd.Series(prices).median())

    best_price = money_to_float(best_flight.get("price")) if best_flight else None
    lowest_price = float(cheapest_offer["price"])

    best_selected_is_lowest = bool(
        best_price is not None
        and round(best_price, 2) == round(lowest_price, 2)
    )

    cheap_prices = [
        money_to_float(offer.get("price"))
        for offer in cheap_flights
    ]
    cheap_prices = [
        price for price in cheap_prices
        if price is not None and price > 0
    ]

    cheap_list_sorted = cheap_prices == sorted(cheap_prices)

    return {
        "available": True,
        "total_live_offers_saved": len(all_offers),
        "valid_numeric_prices": len(valid_offers),
        "price_extraction_rate": round(len(valid_offers) / max(1, len(all_offers)), 3),
        "cheapest_live_price": round(lowest_price, 2),
        "median_live_price": round(median_live_price, 2),
        "savings_vs_median": round(median_live_price - lowest_price, 2),
        "best_selected_price": round(best_price, 2) if best_price is not None else None,
        "best_selected_is_lowest_live_fare": best_selected_is_lowest,
        "cheap_flights_returned": len(cheap_flights),
        "cheap_flights_sorted_by_price": cheap_list_sorted,
        "cheapest_offer": cheapest_offer,
        "best_selected_offer": best_flight,
    }


def evaluate_history_accuracy() -> Dict[str, Any]:
    history = load_json(HISTORY_ANALYSIS_PATH, {})

    if not history:
        return {
            "available": False,
            "message": "No flights.csv history analysis file found. Run the main project script first.",
            "expected_file": str(HISTORY_ANALYSIS_PATH),
        }

    if not history.get("available"):
        return {
            "available": False,
            "message": history.get("message", "Historical analysis was unavailable."),
        }

    required_keys = [
        "records_used",
        "typical_median_fare",
        "best_booking_window",
        "best_search_day",
        "best_travel_day",
        "best_departure_bucket",
        "best_airline",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in history
    ]

    recommendation_fields_present = all(
        history.get(key) is not None
        for key in [
            "best_booking_window",
            "best_search_day",
            "best_travel_day",
            "best_departure_bucket",
        ]
    )

    return {
        "available": True,
        "records_used": history.get("records_used"),
        "scope_level": history.get("scope_level"),
        "scope_note": history.get("scope_note"),
        "typical_median_fare": history.get("typical_median_fare"),
        "missing_expected_keys": missing_keys,
        "recommendation_fields_present": recommendation_fields_present,
        "best_booking_window": history.get("best_booking_window"),
        "best_search_day": history.get("best_search_day"),
        "best_travel_day": history.get("best_travel_day"),
        "best_departure_bucket": history.get("best_departure_bucket"),
        "best_airline": history.get("best_airline"),
    }


def evaluate_final_answer_presence() -> Dict[str, Any]:
    if not FINAL_ANSWER_PATH.exists():
        return {
            "available": False,
            "message": "Final answer text file was not found.",
        }

    text = FINAL_ANSWER_PATH.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    required_sections = [
        "BEST CHEAPEST LIVE FLIGHT",
        "ALL AVAILABLE CHEAP LIVE FLIGHTS",
        "LIVE PRICE SUMMARY",
        "HISTORICAL GUIDANCE FROM flights.csv",
        "LLM-BASED EXPLANATION",
    ]

    missing_sections = [
        section for section in required_sections
        if section not in text
    ]

    return {
        "available": True,
        "character_count": len(text),
        "required_sections_present": len(missing_sections) == 0,
        "missing_sections": missing_sections,
    }


def evaluate_llm_grounding() -> Dict[str, Any]:
    if not FINAL_ANSWER_PATH.exists():
        return {
            "available": False,
            "message": "Final answer text file was not found.",
        }

    final_answer = FINAL_ANSWER_PATH.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    live = evaluate_live_accuracy()
    hist = evaluate_history_accuracy()

    issues = []

    best_offer = live.get("best_selected_offer") or {}
    best_price = live.get("best_selected_price")
    best_airline = best_offer.get("airline")
    best_date = best_offer.get("travel_date")

    if best_price is not None:
        price_text_1 = str(int(best_price))
        price_text_2 = f"{best_price:,.2f}"

        if price_text_1 not in final_answer and price_text_2 not in final_answer:
            issues.append(
                "Final recommendation does not clearly mention the selected cheapest live fare."
            )

    if best_airline and best_airline not in final_answer:
        issues.append(
            "Final recommendation does not clearly mention the selected cheapest airline."
        )

    if best_date and best_date not in final_answer:
        issues.append(
            "Final recommendation does not clearly mention the selected cheapest travel date."
        )

    if hist.get("best_booking_window"):
        value = hist["best_booking_window"].get("booking_window")
        if value and value not in final_answer:
            issues.append(
                "Final recommendation does not mention the best historical booking window."
            )

    if hist.get("best_search_day"):
        value = hist["best_search_day"].get("search_weekday")
        if value and value not in final_answer:
            issues.append(
                "Final recommendation does not mention the best day to search/book."
            )

    if hist.get("best_travel_day"):
        value = hist["best_travel_day"].get("travel_weekday")
        if value and value not in final_answer:
            issues.append(
                "Final recommendation does not mention the best historical travel day."
            )

    if hist.get("best_departure_bucket"):
        value = hist["best_departure_bucket"].get("departure_bucket")
        if value and value not in final_answer:
            issues.append(
                "Final recommendation does not mention the best historical departure time bucket."
            )

    return {
        "available": True,
        "issues_found": len(issues),
        "issues": issues,
        "grounded_recommendation": len(issues) == 0,
    }


def build_report(results: Dict[str, Any]) -> str:
    live = results["live_price_accuracy"]
    hist = results["flights_csv_historical_accuracy"]
    final = results["final_answer_check"]
    llm = results["llm_grounding_accuracy"]

    lines: List[str] = []

    lines.append("╔════════════════════════════════════════════════════════════╗")
    lines.append("║             BudgetFly AI — Output Accuracy Report         ║")
    lines.append("╚════════════════════════════════════════════════════════════╝")
    lines.append("")

    lines.append("1) SERPAPI LIVE PRICE ACCURACY")
    lines.append("------------------------------")

    if live.get("available"):
        rows = [
            ["Total live offers saved", str(live["total_live_offers_saved"])],
            ["Valid numeric prices", str(live["valid_numeric_prices"])],
            ["Price extraction rate", str(live["price_extraction_rate"])],
            ["Cheapest live price", money(live["cheapest_live_price"])],
            ["Median live price", money(live["median_live_price"])],
            ["Savings vs median", money(live["savings_vs_median"])],
            ["Best selected price", money(live["best_selected_price"])],
            ["Best selected is lowest fare", str(live["best_selected_is_lowest_live_fare"])],
            ["Cheap flights returned", str(live["cheap_flights_returned"])],
            ["Cheap list sorted by price", str(live["cheap_flights_sorted_by_price"])],
        ]
        lines.append(table(rows, ["Metric", "Value"]))
    else:
        lines.append(live.get("message", "Live accuracy unavailable."))

    lines.append("")
    lines.append("2) flights.csv HISTORICAL GUIDANCE ACCURACY")
    lines.append("-------------------------------------------")

    if hist.get("available"):
        rows = [
            ["Historical rows used", str(hist.get("records_used"))],
            ["Scope level", str(hist.get("scope_level"))],
            ["Typical median fare", money(hist.get("typical_median_fare"))],
            ["Expected keys missing", ", ".join(hist.get("missing_expected_keys", [])) or "None"],
            ["Guidance fields present", str(hist.get("recommendation_fields_present"))],
        ]

        if hist.get("best_booking_window"):
            item = hist["best_booking_window"]
            rows.append([
                "Best booking window",
                f"{item.get('booking_window')} ({money(item.get('median_fare'))})",
            ])

        if hist.get("best_search_day"):
            item = hist["best_search_day"]
            rows.append([
                "Best search/book day",
                f"{item.get('search_weekday')} ({money(item.get('median_fare'))})",
            ])

        if hist.get("best_travel_day"):
            item = hist["best_travel_day"]
            rows.append([
                "Best travel day",
                f"{item.get('travel_weekday')} ({money(item.get('median_fare'))})",
            ])

        if hist.get("best_departure_bucket"):
            item = hist["best_departure_bucket"]
            rows.append([
                "Best departure time",
                f"{item.get('departure_bucket')} ({money(item.get('median_fare'))})",
            ])

        if hist.get("best_airline"):
            item = hist["best_airline"]
            rows.append([
                "Best airline",
                f"{item.get('airline')} ({money(item.get('median_fare'))})",
            ])

        lines.append(table(rows, ["Metric", "Value"]))
        lines.append(f"Scope note: {hist.get('scope_note')}")
    else:
        lines.append(hist.get("message", "Historical accuracy unavailable."))

    lines.append("")
    lines.append("3) FINAL ANSWER FILE CHECK")
    lines.append("--------------------------")

    if final.get("available"):
        rows = [
            ["Final answer saved", "True"],
            ["Character count", str(final.get("character_count"))],
            ["Required sections present", str(final.get("required_sections_present"))],
            ["Missing sections", ", ".join(final.get("missing_sections", [])) or "None"],
        ]
        lines.append(table(rows, ["Metric", "Value"]))
    else:
        lines.append(final.get("message", "Final answer check unavailable."))

    lines.append("")
    lines.append("4) GPT-5 NANO RECOMMENDATION GROUNDING")
    lines.append("--------------------------------------")

    if llm.get("available"):
        rows = [
            ["Grounded recommendation", str(llm.get("grounded_recommendation"))],
            ["Issues found", str(llm.get("issues_found"))],
        ]
        lines.append(table(rows, ["Metric", "Value"]))

        if llm.get("issues"):
            lines.append("")
            lines.append("Issues:")
            for issue in llm["issues"]:
                lines.append(f"- {issue}")
    else:
        lines.append(llm.get("message", "LLM grounding check unavailable."))

    lines.append("")
    lines.append("5) FINAL VERDICT")
    lines.append("----------------")

    passed = (
        live.get("available")
        and live.get("best_selected_is_lowest_live_fare")
        and live.get("cheap_flights_sorted_by_price")
        and hist.get("available")
        and hist.get("recommendation_fields_present")
        and final.get("available")
        and final.get("required_sections_present")
        and llm.get("available")
        and llm.get("grounded_recommendation")
    )

    if passed:
        lines.append(
            "PASS: The project output is grounded in live SerpApi prices, "
            "flights.csv historical guidance, and the GPT-5 Nano recommendation "
            "mentions the selected live and historical evidence."
        )
    else:
        lines.append(
            "PARTIAL PASS: One or more checks need review before final submission."
        )

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = {
        "live_price_accuracy": evaluate_live_accuracy(),
        "flights_csv_historical_accuracy": evaluate_history_accuracy(),
        "final_answer_check": evaluate_final_answer_presence(),
        "llm_grounding_accuracy": evaluate_llm_grounding(),
    }

    ACCURACY_JSON_PATH.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    report = build_report(results)

    ACCURACY_TEXT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()