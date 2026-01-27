# Daily Diary

This folder contains daily summaries of meaningful training runs, experiments, and insights from the lobe detection project.

## Purpose

- Track progress and learnings over time
- Document what worked and what didn't
- Identify patterns across experiments
- Build foundation for weekly/monthly summaries
- Preserve institutional knowledge

## Structure

Each day gets a markdown file: `YYYY-MM-DD.md`

## Content Guidelines

### What to Include

- **Meaningful runs only** (top ~20% of runs)
  - Runs with significant configuration changes
  - Runs that showed promising results
  - Runs that revealed important failures
  - Runs that tested new hypotheses

- **Key Information:**
  - Run IDs and configurations
  - Results (metrics, outcomes)
  - What worked / what didn't
  - Key insights and learnings
  - Future ideas go to `docs/improvements_backlog.md` (not in diary)

- **Visualizations:**
  - Include plot files when they add value
  - Reference MLflow artifacts
  - Show trends and comparisons

### What to Exclude

- Routine runs with no significant changes
- Runs that duplicate previous experiments
- Minor hyperparameter tweaks with no impact
- Failed runs due to technical errors (not learning issues)

## Format

Each daily entry should include:

1. **Overview** - High-level summary of the day
2. **Key Runs** - Detailed analysis of meaningful runs
3. **What Worked** - Successful approaches/improvements
4. **What Didn't Work** - Failed experiments and why
5. **Key Insights** - Important learnings
6. **Metrics Summary** - Table of key runs
7. **Documentation Created** - New docs/features
8. **Lessons Learned** - Takeaways
9. **Conclusion** - Day's status and outlook

**Note:** Future plans and ideas should go in `docs/improvements_backlog.md`, not in daily diary entries.

## Length

- Target: **200-300 lines** per day
- Focus on quality over quantity
- Be concise but comprehensive

## Weekly/Monthly Summaries

Based on these daily entries, we'll create:
- `weekly/YYYY-WW.md` - Weekly summaries
- `monthly/YYYY-MM.md` - Monthly summaries

These will aggregate insights, identify trends, and provide high-level project status.

## Accessing MLflow Data

To analyze runs for diary entries:
- Use MLflow UI: `poetry run python scripts/start_mlflow_ui.py`
- Check run metadata: `mlruns/<experiment_id>/<run_id>/meta.yaml`
- Review metrics: `mlruns/<experiment_id>/<run_id>/metrics/`
- Check parameters: `mlruns/<experiment_id>/<run_id>/params/`
