"""Streamlit UI for StepSense research prototype."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stepsense.cue import generate_metronome_wav_bytes
from stepsense.movement_model import MovementModel, read_dataset
from stepsense.orchestrator import AlertOrchestrator
from stepsense.stream import replay_recording


def main() -> None:
    st.set_page_config(page_title="StepSense", layout="wide")
    st.title("StepSense - AI Mobility Assistant")
    st.caption(
        "Research prototype for replay-based freezing-of-gait episode analysis with multi-agent review."
    )

    st.warning(
        "Research prototype only. This is not a medical device, is not clinically validated, and "
        "must not be used for diagnosis, fall prediction, or fall prevention decisions."
    )

    model_path = Path("models/model.joblib")
    eval_path = Path("models/eval_results.json")

    with st.sidebar:
        st.header("Controls")
        dataset_path_raw = st.text_input("Dataset path", value="data/daphnet.csv")
        playback_speed = st.slider("Playback speed", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
        detection_threshold = st.slider(
            "Detection threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.05
        )
        consecutive_windows = st.slider("Consecutive windows", min_value=1, max_value=5, value=2, step=1)
        llm_provider = st.selectbox("LLM provider", options=["mock", "openai", "anthropic"], index=0)
        start_monitoring = st.button("Start monitoring", type="primary")

    os.environ["LLM_PROVIDER"] = llm_provider

    live_tab, evaluation_tab = st.tabs(["Live monitor", "Evaluation"])

    with live_tab:
        st.subheader("Live monitor")
        chart_placeholder = st.empty()
        probability_placeholder = st.empty()
        alert_placeholder = st.empty()
        activity_placeholder = st.empty()
        episode_log_placeholder = st.empty()

        model = None
        live_setup_errors: list[str] = []
        if model_path.exists():
            try:
                model = MovementModel.load(model_path)
            except Exception as exc:
                live_setup_errors.append(f"Failed to load model artifact: {exc}")
                st.error(
                    "Model artifact exists but could not be loaded. "
                    "Re-run training to regenerate `models/model.joblib`."
                )
        else:
            st.info(
                "Model file is missing. Run training first:\n\n"
                "`python3 scripts/train.py --dataset /path/to/daphnet.csv`"
            )

        dataset_df: pd.DataFrame | None = None
        dataset_path = Path(dataset_path_raw)
        if dataset_path.exists():
            try:
                dataset_df = read_dataset(dataset_path)
            except Exception as exc:
                live_setup_errors.append(f"Failed to read dataset: {exc}")
                st.error(
                    "Dataset file exists but could not be parsed. "
                    "Verify delimiter/encoding and required columns."
                )
        else:
            st.info(
                "Dataset file not found at the selected path. Update the sidebar path to run replay monitoring."
            )

        selected_participant = None
        if model is not None and dataset_df is not None:
            try:
                participant_column = _resolve_participant_column(dataset_df)
                participant_ids = sorted(dataset_df[participant_column].astype(str).unique().tolist())
                if not participant_ids:
                    st.warning("Dataset has no participant rows to replay.")
                else:
                    default_participant = model.bundle.participant_holdout
                    if default_participant not in participant_ids:
                        default_participant = participant_ids[0]
                    default_index = participant_ids.index(default_participant)
                    selected_participant = st.sidebar.selectbox(
                        "Recording participant",
                        options=participant_ids,
                        index=default_index,
                        help="Defaults to the held-out demo participant from training.",
                    )
            except Exception as exc:
                live_setup_errors.append(f"Failed to prepare participant selector: {exc}")
                st.error(
                    "Unable to prepare replay participant selection. "
                    "Check dataset columns for a participant id field."
                )

        if start_monitoring:
            if model is None or dataset_df is None or selected_participant is None or live_setup_errors:
                st.error("Cannot start monitoring yet. Ensure model and dataset are available.")
            else:
                try:
                    _run_live_monitor(
                        model=model,
                        dataset_df=dataset_df,
                        selected_participant=selected_participant,
                        playback_speed=playback_speed,
                        detection_threshold=detection_threshold,
                        consecutive_windows=consecutive_windows,
                        chart_placeholder=chart_placeholder,
                        probability_placeholder=probability_placeholder,
                        alert_placeholder=alert_placeholder,
                        activity_placeholder=activity_placeholder,
                        episode_log_placeholder=episode_log_placeholder,
                    )
                except Exception as exc:
                    st.error(f"Monitoring run failed: {exc}")

    with evaluation_tab:
        st.subheader("Held-out LOSO evaluation")
        if not eval_path.exists():
            st.info(
                "Evaluation file missing. Run:\n\n"
                "`python3 scripts/train.py --dataset /path/to/daphnet.csv`"
            )
        else:
            try:
                payload = json.loads(eval_path.read_text(encoding="utf-8"))
                normalized = _normalize_eval_payload(payload)
                left, right = st.columns(2)
                with left:
                    st.markdown("**Random Forest mean metrics**")
                    st.json(normalized["rf_mean"])
                with right:
                    st.markdown("**Threshold baseline mean metrics**")
                    st.json(normalized["baseline_mean"])

                st.markdown("**Per-subject Random Forest metrics**")
                st.dataframe(pd.DataFrame(normalized["rf_per_subject"]))
                st.markdown("**Per-subject threshold baseline metrics**")
                st.dataframe(pd.DataFrame(normalized["baseline_per_subject"]))
                st.markdown("**Skipped participants (no freezing in held-out fold)**")
                st.write(normalized["skipped_participants_no_freezing"])
            except json.JSONDecodeError as exc:
                st.error(
                    "Evaluation artifact is invalid JSON. "
                    "Re-run training to regenerate `models/eval_results.json`."
                )
                st.code(str(exc))
            except Exception as exc:
                st.error(
                    "Evaluation artifact has unexpected schema or rendering failed. "
                    "Re-run training to regenerate `models/eval_results.json`."
                )
                st.code(str(exc))

        st.markdown("### Limitations")
        st.markdown(
            "- Research prototype only; not clinically validated.\n"
            "- Model is trained on ankle/thigh/trunk wearable signals from Daphnet-like data.\n"
            "- Phone-only usage is not validated.\n"
            "- Alerts indicate possible freezing episodes and may include false positives/negatives.\n"
            "- Ground truth display is for evaluation only during replay."
        )


def _run_live_monitor(
    *,
    model: MovementModel,
    dataset_df: pd.DataFrame,
    selected_participant: str,
    playback_speed: float,
    detection_threshold: float,
    consecutive_windows: int,
    chart_placeholder,
    probability_placeholder,
    alert_placeholder,
    activity_placeholder,
    episode_log_placeholder,
) -> None:
    participant_column = _resolve_participant_column(dataset_df)
    participant_df = dataset_df[dataset_df[participant_column].astype(str) == selected_participant].copy()
    if participant_df.empty:
        st.error("Selected participant has no rows in the dataset.")
        return

    ankle_column = _resolve_ankle_column(participant_df)
    orchestrator = AlertOrchestrator(consecutive_windows=consecutive_windows, cooldown_seconds=30.0)

    timeline: list[tuple[float, float]] = []
    seen_alert_count = 0
    latest_activity_log = []
    latest_review_payload = None

    stream = replay_recording(participant_df, model, speed=playback_speed, sleep_enabled=False)
    probabilities_over_time: list[float] = []
    times_over_time: list[float] = []
    labels_over_time: list[int] = []
    ankle_points_time: list[float] = []
    ankle_points_value: list[float] = []
    episodes_rows: list[dict[str, float | str]] = []

    for time_seconds, probability, raw_chunk, ground_truth_label in stream:
        timeline.append((time_seconds, probability))
        probabilities_over_time.append(probability)
        times_over_time.append(time_seconds)
        labels_over_time.append(ground_truth_label)

        if ankle_column in raw_chunk.columns:
            step = max(1.0 / model.bundle.sample_rate_hz, 1e-6)
            chunk_times = [time_seconds + index * step for index in range(len(raw_chunk))]
            ankle_points_time.extend(chunk_times)
            ankle_points_value.extend(raw_chunk[ankle_column].astype(float).tolist())

        figure = _build_live_figure(
            ankle_time=ankle_points_time,
            ankle_values=ankle_points_value,
            prob_times=times_over_time,
            probabilities=probabilities_over_time,
            labels=labels_over_time,
        )
        chart_placeholder.plotly_chart(figure, use_container_width=True)
        probability_placeholder.metric("Latest window probability", f"{probability:.3f}")

        alerts = orchestrator.run(
            timeline,
            threshold=detection_threshold,
            false_alarm_cost="medium",
        )
        if len(alerts) > seen_alert_count:
            new_alert = alerts[-1]
            seen_alert_count = len(alerts)
            latest_activity_log = new_alert.activity_log
            latest_review_payload = asdict(new_alert.review)

            cue_bytes = generate_metronome_wav_bytes(bpm=100.0, duration_seconds=4.0)
            alert_placeholder.warning(new_alert.message)
            st.audio(cue_bytes, format="audio/wav", autoplay=True)
            st.markdown("**Review Agent result**")
            st.json(latest_review_payload)

        # Keep this lightweight while still giving visible progression.
        time.sleep(max(0.02, 0.15 / playback_speed))

    final_alerts = orchestrator.run(timeline, threshold=detection_threshold, false_alarm_cost="medium")
    for alert in final_alerts:
        episodes_rows.append(
            {
                "start_time_seconds": round(alert.event.start_time_seconds, 3),
                "duration_seconds": round(alert.event.duration_seconds, 3),
                "mean_confidence": round(alert.event.mean_confidence, 3),
                "urgency": alert.triage.urgency,
                "send_alert": alert.triage.send_alert,
            }
        )
    if final_alerts:
        latest_activity_log = final_alerts[-1].activity_log

    if latest_activity_log:
        activity_rows = [
            {
                "timestamp": item.timestamp,
                "agent": item.agent_name,
                "input": json.dumps(item.input_payload),
                "output": json.dumps(item.output_payload),
            }
            for item in latest_activity_log
        ]
        with activity_placeholder.container():
            st.markdown("### Agent activity panel")
            st.dataframe(pd.DataFrame(activity_rows), use_container_width=True)

    with episode_log_placeholder.container():
        st.markdown("### Episode log")
        if episodes_rows:
            st.dataframe(pd.DataFrame(episodes_rows), use_container_width=True)
        else:
            st.info("No episodes detected with current threshold and settings.")


def _build_live_figure(
    *,
    ankle_time: list[float],
    ankle_values: list[float],
    prob_times: list[float],
    probabilities: list[float],
    labels: list[int],
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ankle_time,
            y=ankle_values,
            name="ankle acceleration",
            mode="lines",
            line={"color": "#1f77b4"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=prob_times,
            y=probabilities,
            name="freezing probability",
            mode="lines",
            yaxis="y2",
            line={"color": "#d62728"},
        )
    )

    for start_time, end_time in _label_segments(prob_times, labels, target_label=1):
        figure.add_vrect(
            x0=start_time,
            x1=end_time,
            fillcolor="rgba(144, 238, 144, 0.2)",
            line_width=0,
            annotation_text="ground truth (for evaluation only)",
            annotation_position="top left",
        )

    figure.update_layout(
        title="Live replay: ankle signal + freezing probability",
        xaxis_title="Time (s)",
        yaxis={"title": "Ankle acceleration"},
        yaxis2={"title": "Freezing probability", "overlaying": "y", "side": "right", "range": [0.0, 1.0]},
        legend={"orientation": "h"},
    )
    return figure


def _label_segments(times: list[float], labels: list[int], target_label: int) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    start_time: float | None = None
    for current_time, label in zip(times, labels):
        if label == target_label and start_time is None:
            start_time = current_time
        elif label != target_label and start_time is not None:
            segments.append((start_time, current_time))
            start_time = None
    if start_time is not None and times:
        segments.append((start_time, times[-1]))
    return segments


def _resolve_participant_column(df: pd.DataFrame) -> str:
    lowered = {column.lower(): column for column in df.columns}
    for candidate in ["subject", "participant", "participant_id", "person", "id"]:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("Unable to resolve participant column.")


def _resolve_ankle_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if "ankle" in column.lower():
            return column
    # Fallback to first numeric non-label column
    lowered = {column.lower(): column for column in df.columns}
    excluded = {lowered.get("label", "label"), lowered.get("time", "time")}
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            return column
    raise ValueError("Unable to resolve ankle signal column for charting.")


def _normalize_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Evaluation payload root must be a JSON object.")

    def _as_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    return {
        "rf_mean": _as_dict(payload.get("rf_mean", {})),
        "baseline_mean": _as_dict(payload.get("baseline_mean", {})),
        "rf_per_subject": _as_list(payload.get("rf_per_subject", [])),
        "baseline_per_subject": _as_list(payload.get("baseline_per_subject", [])),
        "skipped_participants_no_freezing": _as_list(payload.get("skipped_participants_no_freezing", [])),
    }


if __name__ == "__main__":
    main()

