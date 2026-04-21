"""
dag_runner.py
────────────────────────────────────────────────────────────────────
Apache Airflow-style DAG orchestrator for the Social Media
Sentiment & Trend Pipeline.

Concepts covered:
  • Airflow Basics   : DAG, Tasks, dependencies, execution order
  • Airflow Advanced : Retry logic, scheduling, run logs, branching
  • End-to-End       : All pipeline stages wired together
"""

import os
import json
import time
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  AIRFLOW-LIKE CORE ENGINE
# ══════════════════════════════════════════════════════════════════

class TaskStatus(Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    FAILED   = "failed"
    SKIPPED  = "skipped"
    RETRYING = "retrying"


class PipelineTask:
    """Single unit of work in the DAG — mirrors Airflow's BaseOperator."""

    def __init__(self,
                 task_id:          str,
                 python_callable:  Callable,
                 retries:          int   = 2,
                 retry_delay_sec:  float = 1.0,
                 depends_on:       Optional[List[str]] = None,
                 on_failure_skip:  bool  = False):
        self.task_id         = task_id
        self.callable        = python_callable
        self.retries         = retries
        self.retry_delay_sec = retry_delay_sec
        self.depends_on      = depends_on or []
        self.on_failure_skip = on_failure_skip
        self.status          = TaskStatus.PENDING
        self.start_time      = None
        self.end_time        = None
        self.duration_sec    = None
        self.error           = None
        self.attempt         = 0

    def run(self) -> bool:
        self.status     = TaskStatus.RUNNING
        self.start_time = datetime.now()

        for attempt in range(self.retries + 1):
            self.attempt = attempt
            try:
                logger.info(f"Task '{self.task_id}' attempt {attempt+1}/{self.retries+1}")
                self.callable()
                self.status       = TaskStatus.SUCCESS
                self.end_time     = datetime.now()
                self.duration_sec = (self.end_time - self.start_time).total_seconds()
                logger.info(f"Task '{self.task_id}' SUCCESS in {self.duration_sec:.1f}s")
                return True
            except Exception as e:
                self.error = str(e)
                logger.warning(f"Task '{self.task_id}' attempt {attempt+1} failed: {e}")
                if attempt < self.retries:
                    self.status = TaskStatus.RETRYING
                    logger.info(f"  Retrying in {self.retry_delay_sec}s...")
                    time.sleep(self.retry_delay_sec)
                else:
                    self.status       = TaskStatus.FAILED
                    self.end_time     = datetime.now()
                    self.duration_sec = (self.end_time - self.start_time).total_seconds()
                    logger.error(f"Task '{self.task_id}' FAILED: {traceback.format_exc()}")
                    return False
        return False


class DAG:
    """
    Directed Acyclic Graph orchestrator.
    Mirrors Apache Airflow's DAG model with:
      - Topological sort for dependency resolution
      - Dependency failure → downstream skip
      - JSON run log persisted to disk
    """

    def __init__(self, dag_id: str, schedule: str = "@daily",
                 description: str = ""):
        self.dag_id      = dag_id
        self.schedule    = schedule
        self.description = description
        self.tasks:      dict[str, PipelineTask] = {}
        self.created_at  = datetime.now()

    def add_task(self, task: PipelineTask) -> PipelineTask:
        self.tasks[task.task_id] = task
        return task

    def _topological_sort(self) -> List[str]:
        visited, order = set(), []

        def dfs(tid: str):
            if tid in visited:
                return
            visited.add(tid)
            for dep in self.tasks.get(tid, PipelineTask("", lambda: None)).depends_on:
                if dep in self.tasks:
                    dfs(dep)
            order.append(tid)

        for tid in self.tasks:
            dfs(tid)
        return order

    def run(self) -> dict:
        logger.info(f"DAG '{self.dag_id}' started (schedule={self.schedule})")
        exec_start = datetime.now()
        run_id     = f"run_{exec_start.strftime('%Y%m%d_%H%M%S')}"

        run_log = {
            "dag_id":     self.dag_id,
            "run_id":     run_id,
            "started_at": exec_start.isoformat(),
            "schedule":   self.schedule,
            "tasks":      {},
        }

        print(f"\n  {'─'*58}")
        print(f"  DAG: {self.dag_id}")
        print(f"  Run: {run_id}  |  Schedule: {self.schedule}")
        print(f"  {'─'*58}")

        for task_id in self._topological_sort():
            task = self.tasks[task_id]

            # Check dependencies
            dep_ok = all(
                self.tasks[d].status == TaskStatus.SUCCESS
                for d in task.depends_on
                if d in self.tasks
            )
            if not dep_ok:
                task.status = TaskStatus.SKIPPED
                print(f"  ⏭  [{task_id:<25}] SKIPPED (upstream failure)")
                run_log["tasks"][task_id] = {"status": "skipped"}
                continue

            print(f"  ▶  [{task_id:<25}] Running...", end="", flush=True)
            success = task.run()
            icon    = "✅" if success else "❌"
            dur     = f"{task.duration_sec:.1f}s" if task.duration_sec else "?"
            attempts = f"attempt {task.attempt+1}/{task.retries+1}" if task.attempt > 0 else ""
            print(f"\r  {icon} [{task_id:<25}] {task.status.value.upper():10s} {dur:8s} {attempts}")

            run_log["tasks"][task_id] = {
                "status":       task.status.value,
                "duration_sec": round(task.duration_sec or 0, 2),
                "attempts":     task.attempt + 1,
                "error":        task.error,
            }

        # Summary
        total_sec = (datetime.now() - exec_start).total_seconds()
        n_ok  = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SUCCESS)
        n_err = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        n_skip = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)

        run_log.update({
            "finished_at":   datetime.now().isoformat(),
            "total_sec":     round(total_sec, 2),
            "success_count": n_ok,
            "failed_count":  n_err,
            "skipped_count": n_skip,
        })

        print(f"  {'─'*58}")
        print(f"  ✔ {n_ok} succeeded  ✘ {n_err} failed  ⏭ {n_skip} skipped")
        print(f"  Total time: {total_sec:.1f}s")

        # Persist run log
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/dag_{run_id}.json"
        with open(log_file, "w") as f:
            json.dump(run_log, f, indent=2)
        print(f"  Run log → {log_file}")

        return run_log


# ══════════════════════════════════════════════════════════════════
#  TASK CALLABLES
# ══════════════════════════════════════════════════════════════════

def _exec_script(script_name: str):
    abs_path = os.path.abspath(script_name)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Script not found: {abs_path}")
    with open(abs_path) as f:
        code = f.read()
    exec(compile(code, abs_path, "exec"),
         {"__name__": "__main__", "__file__": abs_path})


def task_generate_data():       _exec_script("01_generate_data.py")
def task_etl_pipeline():        _exec_script("02_etl_pipeline.py")
def task_analytics():           _exec_script("03_analytics.py")
def task_streaming():           _exec_script("04_streaming.py")


def task_validate():
    """Final output validation — all expected artefacts must exist."""
    required = [
        "data/warehouse/sentiment_warehouse.db",
        "data/bronze/posts.csv",
        "data/silver/posts.csv",
        "data/gold/topic_sentiment.csv",
        "reports/analytics_dashboard.png",
        "reports/data_quality_report.txt",
        "reports/streaming_report.json",
        "reports/search_index_report.json",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing outputs: {missing}")
    logger.info("Validation passed — all outputs present.")
    print(f"  ✅ All {len(required)} required outputs verified.")


# ══════════════════════════════════════════════════════════════════
#  BUILD THE DAG
# ══════════════════════════════════════════════════════════════════

def build_dag() -> DAG:
    dag = DAG(
        dag_id      = "social_media_sentiment_pipeline",
        schedule    = "@daily",
        description = (
            "End-to-end Social Media Sentiment Pipeline: "
            "data gen → Bronze/Silver/Gold ETL → NLP analytics → streaming → validate"
        ),
    )

    t1 = PipelineTask("generate_data",   task_generate_data,  retries=1)
    t2 = PipelineTask("etl_medallion",   task_etl_pipeline,   retries=2,
                      depends_on=["generate_data"])
    t3 = PipelineTask("nlp_analytics",   task_analytics,      retries=1,
                      depends_on=["etl_medallion"])
    t4 = PipelineTask("kafka_streaming", task_streaming,      retries=1,
                      depends_on=["etl_medallion"])
    t5 = PipelineTask("validate_output", task_validate,       retries=0,
                      depends_on=["nlp_analytics", "kafka_streaming"])

    for t in [t1, t2, t3, t4, t5]:
        dag.add_task(t)

    return dag


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n" + "█" * 60)
    print("  SOCIAL MEDIA SENTIMENT PIPELINE  —  DAG ORCHESTRATOR")
    print("  (Simulated Apache Airflow — Tasks 22, 23, 30)")
    print("█" * 60)

    dag     = build_dag()
    run_log = dag.run()

    print("\n" + "█" * 60)
    status = "✅ PIPELINE COMPLETE" if run_log["failed_count"] == 0 \
             else "⚠  PIPELINE FINISHED WITH ERRORS"
    print(f"  {status}")
    print("█" * 60)
