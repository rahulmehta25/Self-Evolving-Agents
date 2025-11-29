"""
Agent Runner: Executes a specific agent genome against a benchmark task.

This is the M1 version with mocked LLM/tool integration for trace logging.
Full integration will be added in M3.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from ..core.data_store import DataStore
from .trace_logger import TraceLogger


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
    
    M1 Version: Mocked LLM and tool execution for testing trace logging.
    """
    
    def __init__(self, data_store: DataStore):
        """
        Initialize the agent runner.
        
        Args:
            data_store: DataStore instance for persistence
        """
        self.data_store = data_store
    
    def _create_run_manifest(self, genome: Dict[str, Any], task: Dict[str, Any],
                             run_seed: int, generation_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a run manifest for reproducibility."""
        import platform
        import sys
        
        run_id = str(uuid.uuid4())
        
        # Capture environment information
        environment = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version,
            "dependency_hashes": {},  # Will be populated in M3
            "tool_versions": {}  # Will be populated in M3
        }
        
        # LLM config from genome
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
                "system_fingerprint": None  # Will be populated in M3
            },
            "prompt_hashes": {},  # Will be populated in M3
            "benchmark_suite_commit": None  # Will be populated in M3
        }
        
        return manifest
    
    def _mock_llm_call(self, prompt: str, llm_config: Dict[str, Any],
                     trace_logger: TraceLogger, budget_tracker: BudgetTracker) -> str:
        """
        Mock LLM call for M1. Returns a simple response.
        In M3, this will be replaced with actual LLM API calls.
        """
        # Log MODEL_INPUT
        trace_logger.log_event(
            "MODEL_INPUT",
            {"prompt": prompt, "config": llm_config},
            input_content=prompt
        )
        
        # Simulate token usage
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate
        budget_tracker.tokens_used += int(estimated_tokens)
        
        # Mock response (in M3, this will be an actual LLM call)
        if "tool" in prompt.lower() or "use" in prompt.lower():
            response = "I'll use the get_weather tool to find the weather information."
        else:
            response = "I need to think about this problem. Let me provide a final answer: The result is 42."
        
        # Log MODEL_OUTPUT
        trace_logger.log_event(
            "MODEL_OUTPUT",
            {"response": response, "token_usage": {"input": int(estimated_tokens), "output": 50}},
            output_content=response
        )
        
        budget_tracker.tokens_used += 50  # Mock output tokens
        
        return response
    
    def _mock_tool_execute(self, tool_name: str, tool_args: Dict[str, Any],
                          trace_logger: TraceLogger, budget_tracker: BudgetTracker) -> Dict[str, Any]:
        """
        Mock tool execution for M1.
        In M3, this will be replaced with actual tool execution.
        """
        # Log TOOL_CALL
        trace_logger.log_event(
            "TOOL_CALL",
            {"tool_name": tool_name, "arguments": tool_args},
            input_content={"tool_name": tool_name, "arguments": tool_args}
        )
        
        budget_tracker.tool_calls_used += 1
        
        # Mock tool result
        if tool_name == "get_weather":
            result = {"weather": "sunny", "temperature": "15C"}
        elif tool_name == "get_current_time":
            result = {"time": "14:30"}
        else:
            result = {"result": "mock_result"}
        
        # Log TOOL_RESULT
        trace_logger.log_event(
            "TOOL_RESULT",
            {"result": result},
            output_content=result
        )
        
        return result
    
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
    
    def run_task(self, genome: Dict[str, Any], task: Dict[str, Any],
                run_seed: int, generation_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute an agent genome against a benchmark task.
        
        Args:
            genome: Agent genome configuration
            task: Task specification
            run_seed: Random seed for deterministic execution
            generation_id: Optional generation ID
            
        Returns:
            Run result dictionary with run_id, final_response, trace_id, and status
        """
        # 1. Initialization
        manifest = self._create_run_manifest(genome, task, run_seed, generation_id)
        run_id = manifest["run_id"]
        
        # Save initial run record
        self.data_store.save_run(manifest, status="IN_PROGRESS")
        trace_id = self.data_store.get_trace_id(run_id)
        
        if not trace_id:
            raise ValueError(f"Failed to get trace_id for run {run_id}")
        
        trace_logger = TraceLogger(trace_id)
        
        # 2. Setup
        agent_state = AgentState(task["prompt_template"], task.get("context", ""))
        budget_tracker = BudgetTracker(task["budget"])
        
        # Set random seed for deterministic execution
        import random
        random.seed(run_seed)
        
        try:
            # 3. Execution Loop (ReAct-style)
            max_iterations = 10  # Safety limit
            iteration = 0
            
            while not agent_state.is_finished and not budget_tracker.is_exceeded() and iteration < max_iterations:
                iteration += 1
                
                # a. LLM Call
                prompt = agent_state.get_next_prompt()
                llm_response = self._mock_llm_call(
                    prompt,
                    genome.get("llm_config", {}),
                    trace_logger,
                    budget_tracker
                )
                
                # b. Parse Action
                action_type, action_args = self._parse_action(llm_response)
                
                if action_type == "TOOL_CALL":
                    # c. Tool Execution
                    tool_result = self._mock_tool_execute(
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
                    # Continue reasoning
                    agent_state.conversation_history.append({
                        "role": "assistant",
                        "content": llm_response
                    })
            
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
        
        except Exception as e:
            # Log error
            trace_logger.log_event(
                "ERROR",
                {"error": str(e), "error_type": type(e).__name__}
            )
            
            # Save error trace
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
