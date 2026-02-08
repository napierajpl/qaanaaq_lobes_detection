from pathlib import Path
from typing import Optional


def prompt_confirm_seed(
    mismatches: dict,
    csv_path: Path,
    prev_trial_number: str,
) -> bool:
    print("")
    print("WARNING: Previous best hyperparameters may not be compatible with this session.")
    print(f"Source CSV: {csv_path}")
    print(f"Previous best trial number: {prev_trial_number}")
    print("Mismatches:")
    for k, v in mismatches.items():
        print(f"  - {k}: previous={v['previous']}  current={v['current']}")
    print("")
    try:
        answer = input("Use previous best as seed anyway? [y/N]: ").strip().lower()
    except Exception:
        return False
    return answer in ("y", "yes")


def prompt_seed_choice_single_file(
    mismatches: dict,
    recent_best_row: dict,
    compatible_best_row: Optional[dict],
) -> str:
    print("")
    print("WARNING: Most recent session best hyperparameters may be incompatible with this session.")
    print(
        "Most recent session best: "
        f"trial_number={recent_best_row.get('trial_number','?')} value={recent_best_row.get('value','?')} "
        f"session_id={recent_best_row.get('session_id','?')}"
    )
    print("Mismatches:")
    for k, v in mismatches.items():
        print(f"  - {k}: previous={v['previous']}  current={v['current']}")
    print("")
    options: list[tuple[str, str]] = []
    options.append(("1", "Use most recent anyway (may make no sense)"))
    if compatible_best_row is not None:
        options.append(
            (
                "2",
                "Use best compatible from history: "
                f"trial_number={compatible_best_row.get('trial_number','?')} value={compatible_best_row.get('value','?')} "
                f"session_id={compatible_best_row.get('session_id','?')}",
            )
        )
        options.append(("3", "No seeding (cold start)"))
    else:
        options.append(("2", "No seeding (cold start)"))
    print("Choose seeding strategy:")
    for key, label in options:
        print(f"  {key}) {label}")
    try:
        answer = input("Enter choice: ").strip()
    except Exception:
        return "no_seed"
    if compatible_best_row is not None:
        if answer == "1":
            return "use_recent"
        if answer == "2":
            return "use_compatible"
        return "no_seed"
    if answer == "1":
        return "use_recent"
    return "no_seed"
