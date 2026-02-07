"""
ReWOO Agent - Reasoning Without Observation
Three-stage architecture: Plan → Execute → Solve
"""

import os
import ssl
import json
import re
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field

# Use system trust store (macOS Keychain) for SSL certificates
import truststore
truststore.inject_into_ssl()

import certifi
from dotenv import load_dotenv

# Fallback: set certificate bundle paths
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class ToolCall:
   """Represents a single tool call in the plan"""
   id: str
   tool: str
   input: Any


@dataclass
class Evidence:
   """Represents the result of a tool execution"""
   id: str
   tool: str
   input: Any
   output: Any
   status: str  # "success" or "error"
   error_message: str = ""


@dataclass
class Plan:
   """Represents the complete execution plan"""
   description: str
   tool_calls: List[ToolCall]


class ReWOOAgent:
   """
   ReWOO Agent Implementation
  
   Three-stage architecture:
   1. Planner - Creates structured plan with tool calls
   2. Executor - Runs tools and collects evidence
   3. Solver - Analyzes evidence to produce final output
   """
  
   def __init__(self, llm: ChatHuggingFace, tools: Dict[str, Callable] = None):
       self.llm = llm
       self.tools = tools or {}
       self.context: Dict[str, Any] = {}  # Store results for reference resolution
       self.evidence: List[Evidence] = []
      
   def register_tool(self, name: str, func: Callable, description: str = ""):
       """Register a tool for the agent to use"""
       self.tools[name] = {
           "func": func,
           "description": description
       }
  
   def get_tools_description(self) -> str:
       """Generate tools description for the planner prompt"""
       descriptions = []
       for name, tool in self.tools.items():
           desc = tool.get("description", "No description")
           descriptions.append(f"- {name}: {desc}")
       return "\n".join(descriptions)
  
   # ==================== STAGE 1: PLANNER ====================
  
   def plan(self, user_request: str) -> Plan:
       """
       Stage 1: Convert user request into structured JSON plan
       """
       tools_desc = self.get_tools_description()
      
       planner_prompt = f"""<|system|>You are a JSON planning assistant. You MUST output ONLY valid JSON, nothing else.</s>
<|user|>Create a plan for: {user_request}

Available tools:
{tools_desc}

Rules:
- Output ONLY JSON
- Each tool needs id (#E1, #E2), tool name, and input
- Reference previous results with #E1.content or #E1.results

Example JSON:
{{"plan": "Search and summarize", "tool_calls": [{{"id": "#E1", "tool": "search_web", "input": "AI trends"}}, {{"id": "#E2", "tool": "summarize_text", "input": "#E1.content"}}]}}

Now output JSON for the request:</s>
<|assistant|>"""

       response = self.llm.invoke([HumanMessage(content=planner_prompt)])
      
       # Extract JSON from response
       plan_json = self._extract_json(response.content)
      
       if plan_json is None:
           # Fallback: create a default plan based on common patterns
           print("  ⚠️ LLM didn't return valid JSON, using default plan")
           return Plan(
               description="Default search and summarize plan",
               tool_calls=[
                   ToolCall(id="#E1", tool="search_web", input=user_request),
                   ToolCall(id="#E2", tool="summarize_text", input="#E1.content")
               ]
           )
      
       # Parse into Plan object
       tool_calls = [
           ToolCall(
               id=tc.get("id", f"#E{i+1}"),
               tool=tc.get("tool", ""),
               input=tc.get("input", "")
           )
           for i, tc in enumerate(plan_json.get("tool_calls", []))
       ]
      
       return Plan(
           description=plan_json.get("plan", ""),
           tool_calls=tool_calls
       )
  
   def _extract_json(self, text: str) -> Dict:
       """Extract JSON from LLM response"""
       # Try to find JSON in the response
       json_patterns = [
           r'\{[\s\S]*\}',  # Match anything between { and }
       ]
      
       for pattern in json_patterns:
           matches = re.findall(pattern, text)
           for match in matches:
               try:
                   return json.loads(match)
               except json.JSONDecodeError:
                   continue
      
       return None
  
   # ==================== STAGE 2: EXECUTOR ====================
  
   def execute(self, plan: Plan) -> List[Evidence]:
       """
       Stage 2: Execute tools and collect evidence
       """
       self.context = {}
       self.evidence = []
      
       for tool_call in plan.tool_calls:
           evidence = self._execute_tool_call(tool_call)
           self.evidence.append(evidence)
          
           # Store result for reference resolution - handle both string and int IDs
           tool_id = str(tool_call.id).strip("#").replace("E", "")
           # Normalize to E1, E2 format
           self.context[f"E{tool_id}"] = evidence.output
           self.context[tool_id] = evidence.output  # Also store without E prefix
      
       return self.evidence
  
   def _execute_tool_call(self, tool_call: ToolCall) -> Evidence:
       """Execute a single tool call"""
       tool_id = str(tool_call.id)  # Convert to string in case it's an int
       tool_name = tool_call.tool
       tool_input = tool_call.input
      
       # Resolve references in input
       resolved_input = self._resolve_references(tool_input)
      
       # Check if tool exists
       if tool_name not in self.tools:
           return Evidence(
               id=tool_id,
               tool=tool_name,
               input=resolved_input,
               output=None,
               status="error",
               error_message=f"Tool '{tool_name}' not found"
           )
      
       # Execute the tool
       try:
           tool_func = self.tools[tool_name]["func"]
           result = tool_func(resolved_input)
          
           return Evidence(
               id=tool_id,
               tool=tool_name,
               input=resolved_input,
               output=result,
               status="success"
           )
       except Exception as e:
           return Evidence(
               id=tool_id,
               tool=tool_name,
               input=resolved_input,
               output=None,
               status="error",
               error_message=str(e)
           )
  
   def _resolve_references(self, input_value: Any) -> Any:
       """Resolve references like #E1.field to actual values"""
       if isinstance(input_value, str):
           # Check for reference pattern
           ref_pattern = r'#(E\d+)\.?(\w*)'
           match = re.match(ref_pattern, input_value)
          
           if match:
               ref_id = match.group(1)
               field = match.group(2)
              
               if ref_id in self.context:
                   value = self.context[ref_id]
                   if field and isinstance(value, dict):
                       return value.get(field, value)
                   return value
          
           # Handle embedded references
           def replace_ref(m):
               ref_id = m.group(1)
               field = m.group(2)
               if ref_id in self.context:
                   value = self.context[ref_id]
                   if field and isinstance(value, dict):
                       return str(value.get(field, value))
                   return str(value)
               return m.group(0)
          
           return re.sub(ref_pattern, replace_ref, input_value)
      
       elif isinstance(input_value, dict):
           return {k: self._resolve_references(v) for k, v in input_value.items()}
      
       elif isinstance(input_value, list):
           return [self._resolve_references(item) for item in input_value]
      
       return input_value
  
   # ==================== STAGE 3: SOLVER ====================
  
   def solve(self, user_request: str, evidence: List[Evidence]) -> str:
       """
       Stage 3: Analyze evidence and produce final output using LLM
       """
       # Format evidence for the solver
       evidence_text = self._format_evidence(evidence)
      
       solver_prompt = f"""You are a helpful assistant. Based on the gathered evidence, provide a comprehensive answer to the user's request.

USER REQUEST: {user_request}

GATHERED EVIDENCE:
{evidence_text}

INSTRUCTIONS:
1. Analyze all the evidence collected from tool executions
2. Synthesize the information into a clear, helpful response
3. If any tools failed, acknowledge the limitation
4. Provide a direct answer to the user's original request

YOUR RESPONSE:"""

       response = self.llm.invoke([HumanMessage(content=solver_prompt)])
       return response.content
  
   def _format_evidence(self, evidence: List[Evidence]) -> str:
       """Format evidence for the solver prompt"""
       if not evidence:
           return "No evidence collected (no tools were executed)"
      
       formatted = []
       for e in evidence:
           status_icon = "✅" if e.status == "success" else "❌"
           entry = f"""
{status_icon} Tool: {e.tool} (ID: {e.id})
  Input: {e.input}
  Output: {e.output}
  Status: {e.status}"""
           if e.error_message:
               entry += f"\n   Error: {e.error_message}"
           formatted.append(entry)
      
       return "\n".join(formatted)
  
   # ==================== MAIN PIPELINE ====================
  
   def run(self, user_request: str, verbose: bool = True) -> str:
       """
       Run the complete ReWOO pipeline: Plan → Execute → Solve
       """
       if verbose:
           print("=" * 60)
           print("🚀 ReWOO Agent - Starting Pipeline")
           print("=" * 60)
           print(f"\n📝 User Request: {user_request}\n")
      
       # Stage 1: Plan
       if verbose:
           print("-" * 40)
           print("📋 STAGE 1: PLANNING")
           print("-" * 40)
      
       plan = self.plan(user_request)
      
       if verbose:
           print(f"Plan: {plan.description}")
           print(f"Tool Calls: {len(plan.tool_calls)}")
           for tc in plan.tool_calls:
               print(f"  {tc.id}: {tc.tool}({tc.input})")
      
       # Stage 2: Execute
       if verbose:
           print("\n" + "-" * 40)
           print("⚙️  STAGE 2: EXECUTING")
           print("-" * 40)
      
       evidence = self.execute(plan)
      
       if verbose:
           for e in evidence:
               status = "✅" if e.status == "success" else "❌"
               print(f"  {status} {e.id}: {e.tool} → {e.output}")
      
       # Stage 3: Solve
       if verbose:
           print("\n" + "-" * 40)
           print("🧠 STAGE 3: SOLVING")
           print("-" * 40)
      
       result = self.solve(user_request, evidence)
      
       if verbose:
           print("\n" + "=" * 60)
           print("✨ FINAL RESULT")
           print("=" * 60)
      
       return result


# ==================== EXAMPLE TOOLS ====================

def search_web(query: str) -> Dict:
   """Real web search using DuckDuckGo"""
   from duckduckgo_search import DDGS
  
   try:
       # Use verify=False to bypass SSL issues in corporate networks
       with DDGS(verify=False) as ddgs:
           results = list(ddgs.text(query, max_results=5))
      
       if not results:
           return {
               "results": [],
               "content": f"No results found for: {query}"
           }
      
       # Format results
       formatted_results = []
       content_parts = []
      
       for r in results:
           formatted_results.append({
               "title": r.get("title", ""),
               "url": r.get("href", ""),
               "snippet": r.get("body", "")
           })
           content_parts.append(f"{r.get('title', '')}: {r.get('body', '')}")
      
       return {
           "results": formatted_results,
           "content": "\n\n".join(content_parts)
       }
  
   except Exception as e:
       return {
           "results": [],
           "content": f"Search error: {str(e)}",
           "error": str(e)
       }


def summarize_text(text: str) -> Dict:
   """Simulated text summarization tool"""
   # In real implementation, this would use an LLM or summarization model
   if isinstance(text, dict):
       text = str(text)
  
   words = text.split()
   summary = " ".join(words[:20]) + "..." if len(words) > 20 else text
  
   return {
       "summary": summary,
       "word_count": len(words)
   }


def calculate(expression: str) -> Dict:
   """Safe calculator tool"""
   try:
       # Only allow basic math operations
       allowed = set("0123456789+-*/(). ")
       if not all(c in allowed for c in str(expression)):
           return {"error": "Invalid characters in expression", "result": None}
      
       result = eval(str(expression))
       return {"result": result, "expression": expression}
   except Exception as e:
       return {"error": str(e), "result": None}


def get_weather(location: str) -> Dict:
   """Simulated weather tool"""
   # Mock weather data
   return {
       "location": location,
       "temperature": "22°C",
       "condition": "Partly Cloudy",
       "humidity": "65%"
   }


def save_file(data: Any) -> Dict:
   """Simulated file save tool"""
   if isinstance(data, dict):
       content = data.get("content", str(data))
       filename = data.get("filename", "output.txt")
   else:
       content = str(data)
       filename = "output.txt"
  
   # In real implementation, this would save to disk
   return {
       "path": f"/reports/{filename}",
       "size": len(str(content)),
       "status": "saved"
   }


# ==================== MAIN ====================

def main():
   load_dotenv()
  
   # Create LLM - Using zephyr-7b-beta (available on HF Inference API)
   print("🔄 Initializing LLM...")
   endpoint = HuggingFaceEndpoint(
       repo_id="HuggingFaceH4/zephyr-7b-beta",
       task="text-generation",
       max_new_tokens=1024,
       temperature=0.1,  # Lower temperature for more consistent JSON output
   )
   llm = ChatHuggingFace(llm=endpoint)
  
   # Create ReWOO Agent
   agent = ReWOOAgent(llm=llm)
  
   # Register tools
   agent.register_tool(
       "search_web",
       search_web,
       "Search the internet for information (input: search query string)"
   )
   agent.register_tool(
       "summarize_text",
       summarize_text,
       "Summarize text content (input: text to summarize or #E1.content reference)"
   )
   agent.register_tool(
       "calculate",
       calculate,
       "Perform mathematical calculations (input: math expression like '2+2')"
   )
   agent.register_tool(
       "get_weather",
       get_weather,
       "Get weather information (input: location name)"
   )
   agent.register_tool(
       "save_file",
       save_file,
       "Save content to a file (input: {content: 'text', filename: 'name.txt'})"
   )
  
   print("✅ Agent initialized with tools:", list(agent.tools.keys()))
   print()
  
   # Run example query
   user_request = "get weather information in Bengaluru"
   result = agent.run(user_request, verbose=True)
  
   print(f"\n{result}")


if __name__ == "__main__":
   main()


