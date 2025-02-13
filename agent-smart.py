from openai import OpenAI
import inspect
from pydantic import BaseModel
from typing import Optional
import json

client = OpenAI()

class Agent(BaseModel):
    name: str
    model: str = "gpt-4o-mini"
    instructions: str
    tools: list = []

class Response(BaseModel):
    agent: Optional[Agent]
    messages: list

def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    signature = inspect.signature(func)
    parameters = {
        param.name: {"type": type_map.get(param.annotation, "string")}
        for param in signature.parameters.values()
    }

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def route_to_correct_agent(user_message: str):
    """Routes the user to the appropriate agent based on the message content."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyze the user query and determine routing:\n"
                                          "- If the user has a product issue AND wants to buy a new product, return 'issues_then_sales'.\n"
                                          "- If only a product issue is mentioned, return 'issues'.\n"
                                          "- If it's just a purchase request, return 'sales'."},
            {"role": "user", "content": user_message}
        ]
    )

    classification = response.choices[0].message.content.strip().lower()

    if classification == "issues_then_sales":
        return ["issues_and_repairs", "sales"]
    elif classification == "issues":
        return ["issues_and_repairs"]
    else:
        return ["sales"]

def run_full_turn(agent, messages):
    """Executes the agent's response loop until no more tool calls are made."""
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()
    
    while True:
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:
            return Response(agent=current_agent, messages=messages[num_init_messages:])

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

def execute_tool_call(tool_call, tools, agent_name):
    """Executes a function call made by the agent."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")
    return tools[name](**args)

def escalate_to_human(summary):
    """Escalates a case to a human representative."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    exit()

def execute_order(product, price: int):
    """Processes a product order."""
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    return "Order placed successfully."

def execute_refund(item_id, reason="not provided"):
    """Processes a refund for an item."""
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    return "Refund processed successfully."

def look_up_item(search_query):
    """Searches for an item based on a query."""
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right department. "
        "Your priority is to route issues first, before sales."
    ),
    tools=[escalate_to_human],
)

sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc. "
        "Your job is to help customers buy products. "
        "If they mention a product defect, do not process the order until the issue is resolved."
    ),
    tools=[execute_order],
)

issues_and_repairs_agent = Agent(
    name="Issues and Repairs Agent",
    instructions=(
        "You are a customer support agent for ACME Inc. "
        "Handle product issues and process refunds if needed. "
        "Do not transfer the customer until the refund process is completed."
    ),
    tools=[execute_refund, look_up_item],
)

agents_map = {
    "triage": triage_agent,
    "sales": sales_agent,
    "issues_and_repairs": issues_and_repairs_agent
}

agent = triage_agent
messages = []

while True:
    user_input = input("User: ")
    messages.append({"role": "user", "content": user_input})

    agent_sequence = route_to_correct_agent(user_input)

    refund_done = False

    for agent_name in agent_sequence:
        agent = agents_map[agent_name]

        if agent_name == "issues_and_repairs":
            while not refund_done:
                response = run_full_turn(agent, messages)
                messages.extend(response.messages)

                # Check if refund has been processed
                for msg in response.messages:
                    if "refund processed successfully" in msg.get("content", "").lower():
                        refund_done = True
                        print("Issue resolved. Now proceeding with sales.")

        elif agent_name == "sales" and refund_done:
            response = run_full_turn(agent, messages)
            messages.extend(response.messages)