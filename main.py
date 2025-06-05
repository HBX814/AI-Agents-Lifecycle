#!/usr/bin/env python3
"""Complete AI Agent Lifecycle Management System with MCP and Modal GPU Integration"""

import asyncio, json, logging, time, threading, psutil
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import modal
from modal import Image, gpu
import gradio as gr
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app initialization
app = modal.App("mcp-agent-lifecycle")
image = Image.debian_slim().pip_install("transformers", "torch", "accelerate", "bitsandbytes", "psutil")


# ================================
# MCP Protocol Implementation
# ================================

@dataclass
class MCPRequest:
    id: str
    method: str
    params: Dict[str, Any]
    jsonrpc: str = "2.0"


@dataclass
class MCPResponse:
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"


@dataclass
class MCPTool:
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPServer:
    def __init__(self):
        self.tools = {
            "get_metrics": MCPTool("get_metrics", "Get agent metrics",
                                   {"type": "object", "properties": {"agent_id": {"type": "string"}}}),
            "process_text": MCPTool("process_text", "Process text with AI", {"type": "object",
                                                                             "properties": {"text": {"type": "string"},
                                                                                            "agent_id": {
                                                                                                "type": "string"}}}),
            "deploy_agent": MCPTool("deploy_agent", "Deploy new agent",
                                    {"type": "object", "properties": {"config": {"type": "object"}}})
        }

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        try:
            if request.method == "tools/list":
                return MCPResponse(request.id, {"tools": [asdict(tool) for tool in self.tools.values()]})
            elif request.method == "tools/call":
                tool_name = request.params.get("name")
                args = request.params.get("arguments", {})
                result = await self._call_tool(tool_name, args)
                return MCPResponse(request.id, {"content": [{"type": "text", "text": json.dumps(result)}]})
            else:
                return MCPResponse(request.id, error={"code": -32601, "message": "Method not found"})
        except Exception as e:
            return MCPResponse(request.id, error={"code": -32603, "message": str(e)})

    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "get_metrics":
            return lifecycle_manager.get_agent_status(args.get("agent_id"))
        elif tool_name == "process_text":
            return await self._process_with_agent(args.get("text"), args.get("agent_id"))
        elif tool_name == "deploy_agent":
            return lifecycle_manager.deploy_agent(args.get("config", {}))
        return {"error": "Unknown tool"}

    async def _process_with_agent(self, text: str, agent_id: str) -> Dict[str, Any]:
        if agent_id not in lifecycle_manager.agents:
            return {"error": "Agent not found"}
        try:
            agent = lifecycle_manager.agents[agent_id]["instance"]
            result = await agent.process_request.remote.aio(text, {"agent_id": agent_id})
            return result
        except Exception as e:
            return {"error": str(e)}


# ================================
# Modal Agent Classes
# ================================

@app.cls(image=image, gpu=gpu.T4(), timeout=3600, memory=8192)
class OpenSourceAgent:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", load_in_8bit=True)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
        self.metrics = {"requests": 0, "errors": 0, "start_time": time.time()}

    @modal.method()
    async def process_request(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            self.metrics["requests"] += 1
            response = self.generator(f"Human: {text}\nAI:", max_new_tokens=100, do_sample=True, temperature=0.7,
                                      pad_token_id=self.tokenizer.eos_token_id)
            return {"response": response[0]["generated_text"].split("AI:")[-1].strip(),
                    "agent_id": context.get("agent_id", "unknown"), "status": "success"}
        except Exception as e:
            self.metrics["errors"] += 1
            return {"error": str(e), "agent_id": context.get("agent_id", "unknown"), "status": "error"}

    @modal.method()
    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.metrics["start_time"]
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        success_rate = ((self.metrics["requests"] - self.metrics["errors"]) / max(self.metrics["requests"], 1)) * 100
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": round(uptime, 2),
            "cpu_usage": cpu_percent,
            "memory_usage": mem.percent,
            "requests": self.metrics["requests"],
            "errors": self.metrics["errors"],
            "success_rate": round(success_rate, 2)
        }


@app.cls(image=image, gpu=gpu.A10G(), timeout=3600, memory=16384)
class LargeLanguageAgent:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", load_in_4bit=True)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
        self.metrics = {"requests": 0, "errors": 0, "start_time": time.time()}

    @modal.method()
    async def process_request(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            self.metrics["requests"] += 1
            response = self.generator(
                f"Context: {context.get('system_prompt', 'You are a helpful AI assistant.')}\nHuman: {text}\nAI:",
                max_new_tokens=150, do_sample=True, temperature=0.8, pad_token_id=self.tokenizer.eos_token_id)
            return {"response": response[0]["generated_text"].split("AI:")[-1].strip(),
                    "agent_id": context.get("agent_id", "unknown"), "status": "success"}
        except Exception as e:
            self.metrics["errors"] += 1
            return {"error": str(e), "agent_id": context.get("agent_id", "unknown"), "status": "error"}

    @modal.method()
    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.metrics["start_time"]
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        success_rate = ((self.metrics["requests"] - self.metrics["errors"]) / max(self.metrics["requests"], 1)) * 100
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": round(uptime, 2),
            "cpu_usage": cpu_percent,
            "memory_usage": mem.percent,
            "requests": self.metrics["requests"],
            "errors": self.metrics["errors"],
            "success_rate": round(success_rate, 2)
        }


# ================================
# Lifecycle Management
# ================================

class MetricsCollector:
    def __init__(self):
        self.data = {}

    def add_metric(self, agent_id: str, metrics: Dict[str, Any]):
        self.data.setdefault(agent_id, []).append({**metrics, "timestamp": datetime.now()})
        if len(self.data[agent_id]) > 100:  # Keep last 100 metrics
            self.data[agent_id] = self.data[agent_id][-100:]

    def get_latest_metrics(self, agent_id: str) -> Dict[str, Any]:
        return self.data.get(agent_id, [{}])[-1] if self.data.get(agent_id) else {}


class AgentLifecycleManager:
    def __init__(self):
        self.agents = {}
        self.metrics_collector = MetricsCollector()
        self.mcp_server = MCPServer()
        self.monitoring = False

    def deploy_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_id = config.get("agent_id", f"agent_{int(time.time())}")
            agent_type = config.get("type", "standard")

            # Store agent info without actually instantiating Modal class here
            self.agents[agent_id] = {
                "config": config,
                "agent_class": LargeLanguageAgent if agent_type == "large" else OpenSourceAgent,
                "status": "running",
                "deployed_at": datetime.now(),
                "type": agent_type,
                "instance": None  # Will be created when needed
            }

            return {"success": True, "agent_id": agent_id, "status": "deployed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        if agent_id in self.agents:
            self.agents[agent_id]["status"] = "stopped"
            return {"success": True, "message": f"Agent {agent_id} stopped"}
        return {"success": False, "error": "Agent not found"}

    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        if agent_id:
            if agent_id not in self.agents:
                return {"error": "Agent not found"}
            agent_info = self.agents[agent_id].copy()
            agent_info["deployed_at"] = agent_info["deployed_at"].isoformat()
            agent_info["latest_metrics"] = self.metrics_collector.get_latest_metrics(agent_id)
            return agent_info
        else:
            return {
                "total_agents": len(self.agents),
                "running_agents": len([a for a in self.agents.values() if a["status"] == "running"]),
                "agents": {aid: {**data, "deployed_at": data["deployed_at"].isoformat()} for aid, data in
                           self.agents.items()}
            }

    def start_monitoring(self):
        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                for agent_id, agent_data in self.agents.items():
                    if agent_data["status"] == "running":
                        try:
                            # Create instance if not exists
                            if agent_data["instance"] is None:
                                agent_data["instance"] = agent_data["agent_class"]()

                            metrics = agent_data["instance"].get_metrics.remote()
                            self.metrics_collector.add_metric(agent_id, metrics)
                        except Exception as e:
                            logger.error(f"Error collecting metrics for {agent_id}: {e}")
                time.sleep(5)

        threading.Thread(target=monitor_loop, daemon=True).start()

    def stop_monitoring(self):
        self.monitoring = False


# Initialize global manager
lifecycle_manager = AgentLifecycleManager()


# ================================
# Gradio Dashboard
# ================================

def get_system_status():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory().percent
    return f"**System Status:**\n- CPU: {cpu}%\n- Memory: {mem}%\n- Agents: {len(lifecycle_manager.agents)}"


def get_agents_overview():
    rows = []
    for aid, data in lifecycle_manager.agents.items():
        metrics = lifecycle_manager.metrics_collector.get_latest_metrics(aid)
        rows.append({
            "Agent ID": aid,
            "Type": data.get("type", "standard"),
            "Status": data.get("status", "unknown"),
            "Uptime": f"{metrics.get('uptime', 0)}s",
            "CPU %": f"{metrics.get('cpu_usage', 0)}%",
            "Memory %": f"{metrics.get('memory_usage', 0)}%",
            "Requests": metrics.get('requests', 0),
            "Success Rate": f"{metrics.get('success_rate', 0)}%"
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def create_metrics_chart():
    data = []
    for agent_id, metrics_list in lifecycle_manager.metrics_collector.data.items():
        for m in metrics_list[-20:]:  # Last 20 points
            data.append({
                "Agent": agent_id,
                "Time": m.get("timestamp", datetime.now()),
                "CPU": m.get("cpu_usage", 0),
                "Memory": m.get("memory_usage", 0)
            })
    df = pd.DataFrame(data)
    if df.empty:
        return px.line(title="No metrics available")
    return px.line(df, x="Time", y=["CPU", "Memory"], color="Agent", title="Resource Usage Over Time")


def deploy_new_agent(agent_type: str, agent_id: str, system_prompt: str):
    if not agent_id:
        return "❌ Agent ID required", get_agents_overview()
    config = {
        "agent_id": agent_id,
        "type": agent_type,
        "system_prompt": system_prompt or "You are a helpful AI assistant."
    }
    result = lifecycle_manager.deploy_agent(config)
    status = f"✅ Deployed {agent_id}" if result.get("success") else f"❌ Error: {result.get('error')}"
    return status, get_agents_overview()


def chat_with_agent(agent_id: str, message: str, history: List[Dict]):
    if not agent_id or agent_id not in lifecycle_manager.agents:
        history.append({"role": "assistant", "content": "Agent not found or not selected"})
        return history, ""

    try:
        agent_data = lifecycle_manager.agents[agent_id]

        # Create instance if not exists
        if agent_data["instance"] is None:
            agent_data["instance"] = agent_data["agent_class"]()

        result = agent_data["instance"].process_request.remote(message, {"agent_id": agent_id})
        response = result.get("response", result.get("error", "No response"))

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, ""


# Gradio Interface
with gr.Blocks(title="MCP Agent Lifecycle Manager", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 MCP-Enabled AI Agent Lifecycle Manager")

    with gr.Row():
        system_status = gr.Markdown(get_system_status())
        with gr.Column():
            start_monitor_btn = gr.Button("▶️ Start Monitoring", variant="primary")
            stop_monitor_btn = gr.Button("⏹️ Stop Monitoring", variant="secondary")

    with gr.Tabs():
        with gr.TabItem("📊 Dashboard"):
            agents_table = gr.Dataframe(label="Active Agents")
            metrics_chart = gr.Plot(label="Metrics")
            refresh_btn = gr.Button("🔄 Refresh")

        with gr.TabItem("🚀 Deploy"):
            with gr.Row():
                with gr.Column():
                    agent_type_dropdown = gr.Dropdown(["standard", "large"], label="Agent Type", value="standard")
                    new_agent_id = gr.Textbox(label="Agent ID", placeholder="my-agent")
                    system_prompt_input = gr.Textbox(label="System Prompt", placeholder="You are a helpful assistant",
                                                     lines=3)
                    deploy_btn = gr.Button("🚀 Deploy", variant="primary")
                with gr.Column():
                    deploy_status = gr.Textbox(label="Status", interactive=False)
                    updated_table = gr.Dataframe(label="Agents")

        with gr.TabItem("💬 Chat"):
            with gr.Row():
                chat_agent_dropdown = gr.Dropdown(label="Select Agent", choices=[], interactive=True)
                refresh_agents_btn = gr.Button("🔄")
            chatbot = gr.Chatbot(label="Chat with Agent", type="messages")
            msg = gr.Textbox(label="Message", placeholder="Type your message...")
            clear_btn = gr.Button("Clear Chat")

    # Event handlers
    start_monitor_btn.click(lambda: lifecycle_manager.start_monitoring(), outputs=[])
    stop_monitor_btn.click(lambda: lifecycle_manager.stop_monitoring(), outputs=[])

    refresh_btn.click(lambda: (get_agents_overview(), create_metrics_chart(), get_system_status()),
                      outputs=[agents_table, metrics_chart, system_status])

    deploy_btn.click(deploy_new_agent,
                     inputs=[agent_type_dropdown, new_agent_id, system_prompt_input],
                     outputs=[deploy_status, updated_table])

    refresh_agents_btn.click(lambda: gr.Dropdown(choices=list(lifecycle_manager.agents.keys())),
                             outputs=[chat_agent_dropdown])

    msg.submit(chat_with_agent, inputs=[chat_agent_dropdown, msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(lambda: [], outputs=[chatbot])

    # Auto-refresh every 10 seconds
    demo.load(lambda: (get_agents_overview(), create_metrics_chart(), get_system_status()),
              outputs=[agents_table, metrics_chart, system_status])

# ================================
# Main Execution
# ================================

if __name__ == "__main__":
    # Initialize MCP server
    mcp_server = MCPServer()

    # Start the Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )