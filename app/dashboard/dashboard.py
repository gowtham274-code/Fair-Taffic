"""Streamlit faculty dashboard.

STATUS: written for Streamlit/Plotly but NOT executed in the development sandbox (they could not be
installed there). Every chart reads recorded files through ``app.services.results``; if nothing has
been recorded the pages say so -- no sample data is ever shown.

Run:  streamlit run app/dashboard/app.py          (API optional, for starting jobs: FAIRTRAFFIC_API_URL)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from app.services.results import ResultsService
from app.settings import Settings

settings = Settings.from_env()
svc = ResultsService(settings.experiments_root)
API = os.environ.get("FAIRTRAFFIC_API_URL", "http://localhost:8000")
KEY = os.environ.get("FAIRTRAFFIC_API_KEY", "")

st.set_page_config(page_title="Fair Traffic Signal Control via MARL", layout="wide")
PAGES = ["Project Overview", "Traffic Simulation", "AI Agent Monitoring", "Fairness Analysis",
         "Federated Learning", "Performance Comparison", "Experiments and Reports"]
page = st.sidebar.radio("Page", PAGES)
exps = svc.list_experiments()
done = [e for e in exps if e["status"] == "completed"]
st.sidebar.caption(f"{len(done)} completed / {len(exps)} recorded experiments")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, API + path, headers={"X-API-Key": KEY}, timeout=10, **kw)
        return r.status_code, (r.json() if r.content else None)
    except requests.RequestException as e:
        return 0, str(e)


def pick(label="Experiment"):
    if not done:
        st.info("No completed experiment recorded yet. Run one from 'Experiments and Reports' or the CLI.")
        return None
    name = st.selectbox(label, [e["name"] for e in done])
    data = svc.experiment(name)
    if data["config"] and data["config"]["sim"]["backend"] == "mock":
        st.warning("This run used the MOCK queue simulator (test double), not SUMO. Not a research result.")
    return data


if page == PAGES[0]:
    st.title("Fair Traffic Signal Control via Multi-Agent Reinforcement Learning")
    st.subheader("AI-Driven Fair & Adaptive Urban Traffic Management System")
    c1, c2 = st.columns(2)
    c1.markdown("**Problem.** Efficiency-only signal control can leave some approaches waiting far longer than others.")
    c1.markdown("**Solution.** Each intersection is a learning agent; the reward combines waiting time, queues, "
                "throughput and a Generalized Gini Welfare (GGF) fairness term; intersections share *model "
                "updates* (federated averaging), not raw traffic observations.")
    c2.markdown("**Stack.** SUMO + TraCI, Double-DQN agents (NumPy / PyTorch), FedAvg, FastAPI, Streamlit, Plotly, SQLite.")
    c2.markdown("**Implemented vs. planned.** See docs/ARCHITECTURE.md and docs/RESEARCH_LIMITATIONS.md "
                "(e.g. no encryption/differential privacy; pedestrians/cyclists are future work).")
    code, st_ = api("GET", "/api/status")
    if code == 200:
        st.success(f"API reachable. Running job: {st_['running_job']['experiment'] if st_['running_job'] else 'none'}; "
                   f"queued: {st_['queued']}")
    else:
        st.info("API not reachable from the dashboard (read-only mode). Recorded results are still shown.")

elif page == PAGES[1]:
    st.header("Traffic Simulation")
    d = pick()
    if d:
        ev = pd.DataFrame(d["eval_results"])
        st.caption("Per evaluation seed (episode-end metrics of the recorded simulation)")
        st.dataframe(ev, use_container_width=True)
        if not ev.empty:
            cols = [c for c in ["completed_vehicles", "departed_vehicles", "unfinished_vehicles"] if c in ev]
            st.plotly_chart(px.bar(ev, x="seed", y=cols, barmode="group", title="Vehicles departed / completed / unfinished"),
                            use_container_width=True)
        tr = pd.DataFrame(d["agent_trace"])
        if not tr.empty:
            st.plotly_chart(px.line(tr, x="sim_time", y="queue", color="agent", title="Queue per intersection (first evaluation episode)"),
                            use_container_width=True)
            st.plotly_chart(px.scatter(tr, x="sim_time", y="phase", color="agent", title="Signal phase over time"),
                            use_container_width=True)
        st.caption("Live SUMO view: run `sumo_gui` locally (see docs/DEMO_GUIDE.md); the server is headless.")

elif page == PAGES[2]:
    st.header("AI Agent Monitoring")
    d = pick()
    if d:
        tr = pd.DataFrame(d["agent_trace"])
        if tr.empty:
            st.info("This run recorded no agent trace.")
        else:
            agent = st.selectbox("Intersection agent", sorted(tr.agent.unique()))
            a = tr[tr.agent == agent]
            st.plotly_chart(px.line(a, x="sim_time", y=["r_wait", "r_queue", "r_throughput", "r_fairness", "r_total"],
                                    title=f"Reward components of {agent}"), use_container_width=True)
            st.write("Chosen actions (phase index) over time and the observation at each step:")
            st.dataframe(a[["sim_time", "action", "phase", "queue", "obs"]].tail(60), use_container_width=True)
        tl = pd.DataFrame(d["train_log"])
        if not tl.empty:
            st.plotly_chart(px.line(tl, x="episode", y=["reward_total", "reward_wait", "reward_queue", "reward_fairness"],
                                    title="Training reward components per episode"), use_container_width=True)
            st.caption(f"Training episodes: {len(tl)}")
        st.json({k: v for k, v in d["checkpoints"].items()}, expanded=False)

elif page == PAGES[3]:
    st.header("Fairness Analysis")
    d = pick()
    if d:
        ev = pd.DataFrame(d["eval_results"])
        st.markdown("Smaller is better: average / maximum wait, variance, unfinished vehicles, queue. "
                    "Larger is better: Jain index (1 = perfectly equal). GGF score: closer to 0 is better.")
        show = [c for c in ["avg_wait_s", "max_wait_s", "wait_variance", "jain_index", "ggf_score", "avg_queue"] if c in ev]
        st.dataframe(ev[["seed"] + show], use_container_width=True)
        det = d["eval_details"] or []
        if det:
            seed = st.selectbox("Evaluation seed", [x["seed"] for x in det])
            x = next(i for i in det if i["seed"] == seed)
            st.plotly_chart(px.bar(pd.Series(x["intersection_mean_wait_s"], name="s").reset_index(), x="index", y="s",
                                   labels={"index": "intersection", "s": "mean wait (s)"}, title="Mean waiting time by intersection"),
                            use_container_width=True)
            st.plotly_chart(px.bar(pd.Series(x["approach_mean_wait_s"], name="s").reset_index(), x="index", y="s",
                                   labels={"index": "approach", "s": "mean wait (s)"}, title="Mean waiting time by approach"),
                            use_container_width=True)
            st.plotly_chart(px.bar(pd.Series(x["intersection_avg_queue"], name="q").reset_index(), x="index", y="q",
                                   labels={"index": "intersection", "q": "avg queue (veh)"}, title="Average queue by intersection"),
                            use_container_width=True)

elif page == PAGES[4]:
    st.header("Federated Learning")
    fed = svc.latest_federated()
    if not fed:
        st.info("No completed federated experiment recorded yet.")
    else:
        st.write(f"Experiment **{fed['experiment']}** — rounds: {fed['n_rounds']}, global model version: v{fed['global_version']}")
        st.caption("Clients send model weights, a sample count and scalar losses — never observations or replay data. "
                   "All clients run in one process (logical FL); no encryption is implemented.")
        rd = pd.DataFrame([{"round": r["round"], "status": r["status"], "version": r["global_version"],
                            "participants": len(r["participants"]), "rejected": len(r["rejected"]),
                            "update_l2": r["mean_update_l2"], "bytes_up": r["bytes_up"], "bytes_down": r["bytes_down"]}
                           for r in fed["rounds"]])
        st.dataframe(rd, use_container_width=True)
        st.plotly_chart(px.line(rd, x="round", y="update_l2", title="Mean client-update L2 distance per round"), use_container_width=True)
        st.write("Global checkpoint history")
        st.dataframe(pd.DataFrame(fed["checkpoints"]), use_container_width=True)

elif page == PAGES[5]:
    st.header("Performance Comparison")
    comp = pd.DataFrame(svc.comparison())
    if comp.empty:
        st.info("No completed experiments to compare yet (run `python -m experiments.run_suite`).")
    else:
        if (comp.backend == "mock").any():
            st.warning("Rows with backend 'mock' come from a test double, not SUMO.")
        backend = st.selectbox("Backend", sorted(comp.backend.unique()))
        c = comp[comp.backend == backend]
        metric = st.selectbox("Metric", ["avg_wait_s", "max_wait_s", "jain_index", "ggf_score", "throughput_veh_per_h",
                                         "avg_queue", "wait_variance", "unfinished_vehicles"])
        st.plotly_chart(px.bar(c, x="scenario", y=f"{metric}_mean", error_y=f"{metric}_std", color="variant", barmode="group",
                               title=metric), use_container_width=True)
        st.dataframe(c, use_container_width=True)
        st.caption("Mean ± std pooled over runs and evaluation seeds (identical demand seeds for all controllers).")

else:
    st.header("Experiments and Reports")
    if exps:
        st.dataframe(pd.DataFrame(exps).drop(columns=["path"], errors="ignore"), use_container_width=True)
    d = pick("Inspect experiment")
    if d:
        st.json(d["config"], expanded=False)
        st.json(d["summary"]["meta"] if d["summary"] else {}, expanded=False)
        for f in ("eval_results.csv", "train_log.csv", "agent_trace.csv"):
            p = svc.csv_path(d["name"], f)
            if p:
                st.download_button(f"Download {f}", Path(p).read_bytes(), file_name=f"{d['name']}_{f}")
    st.subheader("Start an experiment (needs the API and an admin key)")
    with st.form("run"):
        kind = st.selectbox("Kind", ["baseline", "marl", "centralized", "federated"])
        scenario = st.selectbox("Scenario", ["low", "medium", "high", "uneven"])
        backend = st.selectbox("Backend", ["sumo", "mock"])
        episodes = st.number_input("Training episodes", 1, 300, 10)
        if st.form_submit_button("Submit job"):
            code, resp = api("POST", "/api/experiments/run",
                             json={"kind": kind, "scenario": scenario, "backend": backend, "episodes": int(episodes)})
            (st.success if code == 202 else st.error)(f"HTTP {code}: {resp}")
    code, jobs = api("GET", "/api/jobs")
    if code == 200 and jobs:
        st.dataframe(pd.DataFrame(jobs).drop(columns=["progress"], errors="ignore"), use_container_width=True)
