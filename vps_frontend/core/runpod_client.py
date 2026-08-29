"""
SupoClip — RunPod Serverless API Client
Handles job submission, polling, and result retrieval from RunPod endpoints.
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"

# Job terminal states
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
IN_PROGRESS_STATES = {"IN_QUEUE", "IN_PROGRESS"}


class RunPodClient:
    """
    Client for RunPod Serverless Endpoints.
    
    Handles async job submission with polling until completion.
    """

    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = (api_key or "").strip()
        self.endpoint_id = (endpoint_id or "").strip()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.base = f"{RUNPOD_BASE_URL}/{self.endpoint_id}"
        logger.info(f"RunPodClient initialized | endpoint={self.endpoint_id}")

    def submit_job(self, payload: dict[str, Any]) -> str:
        """
        Submit a job to the RunPod serverless endpoint.

        Args:
            payload: Dict with 'input' key containing job parameters.

        Returns:
            job_id: String identifier for the submitted job.

        Raises:
            RuntimeError: If submission fails after retries.
        """
        url = f"{self.base}/run"
        body = {"input": payload}

        for attempt in range(1, 4):
            try:
                resp = requests.post(url, json=body, headers=self.headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                job_id = data.get("id")
                if not job_id:
                    raise RuntimeError(f"No job ID in response: {data}")
                logger.info(f"Job submitted | job_id={job_id}")
                return job_id

            except requests.HTTPError as e:
                logger.error(f"HTTP error submitting job (attempt {attempt}): {e}")
                if attempt == 3:
                    raise RuntimeError(f"RunPod job submission failed: {e}") from e
                time.sleep(2 ** attempt)

            except requests.RequestException as e:
                logger.error(f"Request error (attempt {attempt}): {e}")
                if attempt == 3:
                    raise RuntimeError(f"RunPod submission request failed: {e}") from e
                time.sleep(2 ** attempt)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Fetch current status of a RunPod job.

        Returns:
            Dict with 'status', 'output', 'error' keys.
        """
        url = f"{self.base}/status/{job_id}"
        resp = requests.get(url, headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def poll_until_complete(
        self,
        job_id: str,
        poll_interval: int = 10,
        timeout: int = 7200,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Block until the job reaches a terminal state.

        Args:
            job_id: The RunPod job ID.
            poll_interval: Seconds between status checks (default 10).
            timeout: Max seconds to wait (default 7200 = 2 hours).
            progress_callback: Optional callable(status_str) for UI updates.

        Returns:
            The final job result dict.

        Raises:
            TimeoutError: If the job doesn't complete within timeout.
            RuntimeError: If the job fails or is cancelled.
        """
        start_time = time.time()
        last_status = None

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Job {job_id} timed out after {timeout}s"
                )

            try:
                data = self.get_job_status(job_id)
                status = data.get("status", "UNKNOWN")

                if status != last_status:
                    logger.info(f"Job {job_id} | status={status} | elapsed={elapsed:.0f}s")
                    if progress_callback:
                        progress_callback(status)
                    last_status = status

                if status in TERMINAL_STATES:
                    if status == "COMPLETED":
                        return data
                    else:
                        error_msg = data.get("error", f"Job ended with status: {status}")
                        raise RuntimeError(
                            f"RunPod job {job_id} failed: {error_msg}"
                        )

            except requests.RequestException as e:
                logger.warning(f"Polling error for {job_id}: {e}. Retrying in {poll_interval}s...")

            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job. Returns True on success."""
        try:
            url = f"{self.base}/cancel/{job_id}"
            resp = requests.post(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            logger.info(f"Job {job_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def check_health(self) -> dict[str, Any]:
        """Check the health/capacity of the endpoint."""
        try:
            url = f"{self.base}/health"
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": str(e)}
