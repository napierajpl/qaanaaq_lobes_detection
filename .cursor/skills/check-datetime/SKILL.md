---
name: check-datetime
description: Get current date and time by running a shell command. Use whenever the agent needs to know the current date, time, or timestamp (e.g. diary entries, endday, logging, filenames).
---

# Check date/time

## When to use

Use this skill whenever you need the **current** date or time. Do not rely on "Today's date" from user_info alone for critical uses (e.g. creating or naming diary entries); run the check and use the result.

Typical cases:
- Writing a daily diary entry (file name and heading date)
- Endday command (which day to write the entry for)
- Any log message or artifact that should be timestamped
- Filenames or headings that include the date

## How to get current date/time

Run a shell command. Prefer **bash** (works in Git Bash on Windows and on Linux/macOS).

**Full timestamp (date + time):**
```bash
date '+%Y-%m-%d %H:%M:%S %Z'
```
Example: `2026-02-16 14:32:01 GMT`

**Date only (for diary filenames like 2026-02-16.md):**
```bash
date '+%Y-%m-%d'
```

**Human-readable date (for diary headings):**
```bash
date '+%A %B %d, %Y'
```
Example: `Monday February 16, 2026`

**If `date` is not available**, try:
- PowerShell: `Get-Date -Format "yyyy-MM-dd HH:mm:ss"`
- Python one-liner: `python -c "from datetime import datetime; print(datetime.now().isoformat())"`

## Workflow

1. When a task requires the current date or time, run one of the commands above in the project root (or current workspace).
2. Use the command output as the authoritative current date/time for that task.
3. Do not assume or reuse a date from earlier in the conversation for time-sensitive output (e.g. diary entry for "today").
