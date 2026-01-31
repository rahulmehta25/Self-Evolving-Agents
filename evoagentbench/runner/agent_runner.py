"""
Agent Runner: Executes a specific agent genome against a benchmark task.

Supports mock LLM (default) and Vertex AI Gemini. Set EVOAGENTBENCH_LLM=vertex
to use real Gemini via GCP; set EVOAGENTBENCH_GCP_PROJECT if needed.

Guide §5.2: Thread-based timeout for time budget enforcement.
"""

import hashlib
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.data_store import DataStore
from .trace_logger import TraceLogger
from . import llm_adapters
from .llm_adapters import ExternalFailureError
from . import tool_adapters


class BudgetTracker:
    """Tracks budget constraints for agent execution."""
    
    def __init__(self, budget: Dict[str, int]):
        self.max_tokens = budget.get("max_tokens", 10000)
        self.max_tool_calls = budget.get("max_tool_calls", 100)
        self.max_time_seconds = budget.get("max_time_seconds", 300)
        
        self.tokens_used = 0
        self.tool_calls_used = 0
        self.start_time = time.time()
    
    def is_exceeded(self) -> bool:
        """Check if any budget has been exceeded."""
        return (
            self.tokens_used >= self.max_tokens or
            self.tool_calls_used >= self.max_tool_calls or
            (time.time() - self.start_time) >= self.max_time_seconds
        )
    
    def get_status(self) -> str:
        """Get the current budget status."""
        if self.tokens_used >= self.max_tokens:
            return "BUDGET_EXCEEDED"
        if self.tool_calls_used >= self.max_tool_calls:
            return "BUDGET_EXCEEDED"
        if (time.time() - self.start_time) >= self.max_time_seconds:
            return "TIMEOUT"
        return "IN_PROGRESS"


class AgentState:
    """Tracks the internal state of the agent during execution."""
    
    def __init__(self, prompt_template: str, context: str = ""):
        self.prompt_template = prompt_template
        self.context = context
        self.is_finished = False
        self.final_answer: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.tool_results: List[Dict[str, Any]] = []
    
    def get_next_prompt(self) -> str:
        """Get the next prompt to send to the LLM."""
        # Simple implementation: combine template, context, and history
        prompt = self.prompt_template
        if self.context:
            prompt = f"{self.context}\n\n{prompt}"
        
        # Add conversation history if available
        if self.conversation_history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history[-5:]  # Last 5 messages
            ])
            prompt = f"{prompt}\n\nPrevious conversation:\n{history_text}"
        
        return prompt
    
    def update_with_tool_result(self, tool_result: Dict[str, Any]):
        """Update state with a tool execution result."""
        self.tool_results.append(tool_result)
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Tool result: {tool_result}"
        })
    
    def set_final_answer(self, answer: str):
        """Set the final answer and mark as finished."""
        self.final_answer = answer
        self.is_finished = True
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Final Answer: {answer}"
        })


class AgentRunner:
    """
    Executes an agent genome against a benchmark task.
    
    Uses mock LLM by default. Set llm_provider="vertex" or EVOAGENTBENCH_LLM=vertex
    to use Vertex AI Gemini (requires google-cloud-aiplatform and ADC).
    """
    
    def __init__(self, data_store: DataStore, llm_provider: Optional[str] = None):
        """
        Initialize the agent runner.
        
        Args:
            data_store: DataStore instance for persistence
            llm_provider: "vertex" for Vertex AI Gemini, "mock" for mock. 
                Defaults to env EVOAGENTBENCH_LLM or "mock".
        """
        self.data_store = data_store
        raw = (llm_provider or os.environ.get("EVOAGENTBENCH_LLM") or "").strip().lower()
        if raw in ("vertex", "mock"):
            self.llm_provider = raw
        else:
            self.llm_provider = llm_adapters.get_llm_provider()
    
    def _create_run_manifest(
        self,
        genome: Dict[str, Any],
        task: Dict[str, Any],
        run_seed: int,
        generation_id: Optional[int] = None,
        *,
        system_prompt: str = "",
        user_prompt: str = "",
        benchmark_suite_commit: Optional[str] = None,
        dependency_hashes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a run manifest for reproducibility (Guide §6.1)."""
        import platform
        import sys

        run_id = str(uuid.uuid4())
        system_prompt = system_prompt or genome.get("system_prompt", "")
        user_prompt = user_prompt or ""

        prompt_hashes: Dict[str, str] = {}
        if system_prompt:
            prompt_hashes["system_prompt"] = hashlib.sha256(system_prompt.encode()).hexdigest()
        if user_prompt:
            prompt_hashes["user_prompt"] = hashlib.sha256(user_prompt.encode()).hexdigest()

        deps = dependency_hashes if dependency_hashes is not None else {}
        environment = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version,
            "dependency_hashes": deps,
            "tool_versions": {}
        }

        llm_config = genome.get("llm_config", {})
        manifest = {
            "run_id": run_id,
            "task_id": task["task_id"],
            "task_version": task["version"],
            "genome_id": genome["genome_id"],
            "generation_id": generation_id,
            "run_seed": run_seed,
            "start_timestamp": datetime.utcnow().isoformat(),
            "environment": environment,
            "llm_config": {
                "model_name": llm_config.get("model_name", "mock-model"),
                "model_version": llm_config.get("model_version", "v1.0"),
                "temperature": llm_config.get("temperature", 0.0),
                "top_p": llm_config.get("top_p", 1.0),
                "seed": run_seed,
                "system_fingerprint": None
            },
            "prompt_hashes": prompt_hashes,
            "benchmark_suite_commit": benchmark_suite_commit,
        }
        return manifest
    
    def _do_llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        llm_config: Dict[str, Any],
        run_seed: int,
        trace_logger: TraceLogger,
        budget_tracker: BudgetTracker,
    ) -> str:
        """Dispatch to Vertex Gemini or mock LLM."""
        if self.llm_provider == "vertex" and llm_adapters._vertex_gemini_available():
            return self._vertex_llm_call(
                system_prompt, user_prompt, llm_config, run_seed, trace_logger, budget_tracker
            )
        return self._mock_llm_call(user_prompt, llm_config, trace_logger, budget_tracker)
    
    def _vertex_llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        llm_config: Dict[str, Any],
        run_seed: int,
        trace_logger: TraceLogger,
        budget_tracker: BudgetTracker,
    ) -> str:
        """Call Vertex AI Gemini and log trace / update budget."""
        trace_logger.log_event(
            "MODEL_INPUT",
            {"prompt": user_prompt, "system_prompt": system_prompt[:200], "config": llm_config},
            input_content=user_prompt,
        )
        try:
            text, inp, out = llm_adapters.call_vertex_gemini(
                system_prompt, user_prompt, llm_config, run_seed
            )
        except Exception as e:
            trace_logger.log_event(
                "MODEL_OUTPUT",
                {"response": "", "error": str(e), "token_usage": {"input": 0, "output": 0}},
                output_content="",
            )
            raise
        budget_tracker.tokens_used += inp + out
        trace_logger.log_event(
            "MODEL_OUTPUT",
            {"response": text, "token_usage": {"input": inp, "output": out}},
            output_content=text,
        )
        return text
    
    def _mock_llm_call(self, prompt: str, llm_config: Dict[str, Any],
                     trace_logger: TraceLogger, budget_tracker: BudgetTracker) -> str:
        """
        Mock LLM call. Returns a simple response when Vertex is not in use.
        """
        trace_logger.log_event(
            "MODEL_INPUT",
            {"prompt": prompt, "config": llm_config},
            input_content=prompt
        )
        estimated_tokens = len(prompt.split()) * 1.3
        budget_tracker.tokens_used += int(estimated_tokens)
        if "tool" in prompt.lower() or "use" in prompt.lower():
            response = "I'll use the get_weather tool to find the weather information."
        else:
            response = "I need to think about this problem. Let me provide a final answer: The result is 42."
        trace_logger.log_event(
            "MODEL_OUTPUT",
            {"response": response, "token_usage": {"input": int(estimated_tokens), "output": 50}},
            output_content=response
        )
        budget_tracker.tokens_used += 50
        return response
    
    def _do_tool_execute(self, tool_name: str, tool_args: Dict[str, Any],
                         trace_logger: TraceLogger, budget_tracker: BudgetTracker) -> Dict[str, Any]:
        """
        Execute tool: Vertex Search for retrieval/search, real URL if configured, else mock.
        """
        trace_logger.log_event(
            "TOOL_CALL",
            {"tool_name": tool_name, "arguments": tool_args},
            input_content={"tool_name": tool_name, "arguments": tool_args}
        )
        budget_tracker.tool_calls_used += 1

        result: Dict[str, Any] = {}
        if tool_name in ("retrieval", "search") and tool_adapters.vertex_search_available():
            query = (tool_args.get("query") or tool_args.get("q") or "").strip() or "general"
            hits = tool_adapters.call_vertex_search(query)
            result = {"results": hits, "count": len(hits)}
        elif tool_adapters.get_tool_url(tool_name):
            result = tool_adapters.call_real_tool_url(tool_name, tool_args)
            if not result:
                result = {"error": "Tool URL call failed"}
        else:
            if tool_name == "get_weather":
                result = {"weather": "sunny", "temperature": "15C"}
            elif tool_name == "get_current_time":
                result = {"time": "14:30"}
            else:
                result = {"result": "mock_result"}

        trace_logger.log_event(
            "TOOL_RESULT",
            {"result": result},
            output_content=result
        )
        return result

    def _mock_tool_execute(self, tool_name: str, tool_args: Dict[str, Any],
                          trace_logger: TraceLogger, budget_tracker: BudgetTracker) -> Dict[str, Any]:
        """Legacy name: delegates to _do_tool_execute."""
        return self._do_tool_execute(tool_name, tool_args, trace_logger, budget_tracker)
    
    def _parse_action(self, llm_response: str) -> tuple:
        """
        Parse the LLM response to extract action type and arguments.
        Simple implementation for M1.
        """
        response_lower = llm_response.lower()
        
        if "final answer" in response_lower or "answer:" in response_lower:
            # Extract answer
            if ":" in llm_response:
                answer = llm_response.split(":", 1)[1].strip()
            else:
                answer = llm_response
            return "FINAL_ANSWER", {"answer": answer}
        
        elif "tool" in response_lower or "use" in response_lower:
            # Mock tool call
            if "weather" in response_lower:
                return "TOOL_CALL", {"tool_name": "get_weather", "arguments": {"city": "London, UK"}}
            elif "time" in response_lower:
                return "TOOL_CALL", {"tool_name": "get_current_time", "arguments": {"city": "London, UK"}}
            else:
                return "TOOL_CALL", {"tool_name": "mock_tool", "arguments": {}}
        
        else:
            return "CONTINUE", {}
    
    def _execute_agent_loop(
        self,
        genome: Dict[str, Any],
        agent_state: AgentState,
        budget_tracker: BudgetTracker,
        trace_logger: TraceLogger,
        run_seed: int,
    ) -> None:
        """
        Execute the ReAct-style agent loop. Modifies agent_state in place.
        Separated for thread-based timeout (Guide §5.2).
        """
        max_iterations = 10
        iteration = 0

        while not agent_state.is_finished and not budget_tracker.is_exceeded() and iteration < max_iterations:
            iteration += 1

            user_prompt = agent_state.get_next_prompt()
            system_prompt = genome.get("system_prompt", "")
            llm_response = self._do_llm_call(
                system_prompt,
                user_prompt,
                genome.get("llm_config", {}),
                run_seed,
                trace_logger,
                budget_tracker,
            )

            action_type, action_args = self._parse_action(llm_response)

            if action_type == "TOOL_CALL":
                tool_result = self._do_tool_execute(
                    action_args["tool_name"],
                    action_args.get("arguments", {}),
                    trace_logger,
                    budget_tracker
                )
                agent_state.update_with_tool_result(tool_result)
            elif action_type == "FINAL_ANSWER":
                agent_state.set_final_answer(action_args.get("answer", llm_response))
                break
            else:
                agent_state.conversation_history.append({
                    "role": "assistant",
                    "content": llm_response
                })

    def run_task(
        self,
        genome: Dict[str, Any],
        task: Dict[str, Any],
        run_seed: int,
        generation_id: Optional[int] = None,
        *,
        benchmark_suite_commit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agent genome against a benchmark task.

        Args:
            genome: Agent genome configuration
            task: Task specification
            run_seed: Random seed for deterministic execution
            generation_id: Optional generation ID
            benchmark_suite_commit: Optional git commit (or tag) of the benchmark suite (Guide §4.5).

        Returns:
            Run result dictionary with run_id, final_response, trace_id, and status
        """
        # Build user prompt from prompt_template + input_params (Guide §4.3)
        prompt_template = task["prompt_template"]
        input_params = task.get("input_params") or {}
        try:
            formatted_template = prompt_template.format_map(defaultdict(str, input_params))
        except KeyError:
            formatted_template = prompt_template
        context = task.get("context", "")
        user_prompt = f"{context}\n\n{formatted_template}".strip() if context else formatted_template
        system_prompt = genome.get("system_prompt", "")

        # 1. Initialization
        manifest = self._create_run_manifest(
            genome, task, run_seed, generation_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            benchmark_suite_commit=benchmark_suite_commit,
        )
        run_id = manifest["run_id"]

        # Save initial run record
        self.data_store.save_run(manifest, status="IN_PROGRESS")
        trace_id = self.data_store.get_trace_id(run_id)

        if not trace_id:
            raise ValueError(f"Failed to get trace_id for run {run_id}")

        trace_logger = TraceLogger(trace_id)

        # 2. Setup (use formatted template so input_params are applied)
        agent_state = AgentState(formatted_template, context)
        budget_tracker = BudgetTracker(task["budget"])
        time_budget = budget_tracker.max_time_seconds

        # Set random seed for deterministic execution (Guide §5.1)
        import random
        random.seed(run_seed)
        try:
            import numpy as np
            np.random.seed(run_seed)
        except Exception:
            pass

        try:
            # 3. Execution Loop with thread-based timeout (Guide §5.2)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._execute_agent_loop,
                    genome,
                    agent_state,
                    budget_tracker,
                    trace_logger,
                    run_seed,
                )
                try:
                    future.result(timeout=time_budget)
                except FuturesTimeoutError:
                    # Thread timeout - mark as TIMEOUT status
                    trace_logger.log_event(
                        "TIMEOUT",
                        {"message": f"Execution exceeded {time_budget}s time budget"}
                    )
                    agent_state.final_answer = "Execution timed out."

            # 4. Finalization
            if not agent_state.final_answer:
                agent_state.final_answer = "No final answer provided."

            # Log final answer
            trace_logger.log_event(
                "FINAL_ANSWER",
                {"answer": agent_state.final_answer}
            )

            # Save trace events to data store
            for event in trace_logger.get_trace():
                self.data_store.save_trace_event(
                    trace_id=event["trace_id"],
                    step_index=event["step_index"],
                    timestamp=event["timestamp"],
                    event_type=event["event_type"],
                    payload=event["payload"],
                    input_hash=event.get("input_hash"),
                    output_hash=event.get("output_hash")
                )

            # Update run status
            status = budget_tracker.get_status()
            if status == "IN_PROGRESS":
                status = "SUCCESS"

            self.data_store.update_run_status(
                run_id,
                status,
                end_timestamp=datetime.utcnow()
            )

            return {
                "run_id": run_id,
                "trace_id": trace_id,
                "final_response": agent_state.final_answer,
                "status": status,
                "tokens_used": budget_tracker.tokens_used,
                "tool_calls_used": budget_tracker.tool_calls_used
            }

        except ExternalFailureError as e:
            # Guide §5.2: external call failed after retries
            trace_logger.log_event(
                "ERROR",
                {"error": str(e), "error_type": "ExternalFailure"}
            )
            for event in trace_logger.get_trace():
                self.data_store.save_trace_event(
                    trace_id=event["trace_id"],
                    step_index=event["step_index"],
                    timestamp=event["timestamp"],
                    event_type=event["event_type"],
                    payload=event["payload"],
                    input_hash=event.get("input_hash"),
                    output_hash=event.get("output_hash")
                )
            self.data_store.update_run_status(run_id, "EXTERNAL_FAILURE", datetime.utcnow())
            return {
                "run_id": run_id,
                "trace_id": trace_id,
                "final_response": None,
                "status": "EXTERNAL_FAILURE",
                "error": str(e)
            }
        except Exception as e:
            trace_logger.log_event(
                "ERROR",
                {"error": str(e), "error_type": type(e).__name__}
            )
            for event in trace_logger.get_trace():
                self.data_store.save_trace_event(
                    trace_id=event["trace_id"],
                    step_index=event["step_index"],
                    timestamp=event["timestamp"],
                    event_type=event["event_type"],
                    payload=event["payload"],
                    input_hash=event.get("input_hash"),
                    output_hash=event.get("output_hash")
                )
            self.data_store.update_run_status(run_id, "FAILURE", datetime.utcnow())
            return {
                "run_id": run_id,
                "trace_id": trace_id,
                "final_response": None,
                "status": "FAILURE",
                "error": str(e)
            }
