from typing import Tuple

def format_metric(value: float, metric_name: str) -> Tuple[float, str]:
    if value is None:
        return 0.0, "N/A"
    
    # Define thresholds for each metric
    thresholds = {
        "sharpe": [(2.0, "Excellent"), (1.0, "Good"), (0.5, "Modest"), (0.0, "Poor")],
        "sortino": [(2.0, "Excellent"), (1.5, "Good"), (1.0, "Modest"), (0.0, "Poor")],
        "calmar": [(1.0, "Excellent"), (0.5, "Fair"), (0.3, "Weak"), (0.0, "Bad")]
    }

    levels = thresholds.get(metric_name.lower(), [])
    explanation = "N/A"
    for thresh, label in levels:
        if value >= thresh:
            explanation = label
            break
    else:
        explanation = "Negative"

    # Positive delta if metric is good (green), negative if bad (red)
    delta_value = 1.0 if value >= levels[1][0] else -1.0
    delta_text = f"{explanation}"

    return round(value, 2), delta_text if delta_value > 0 else f"-{explanation}"
