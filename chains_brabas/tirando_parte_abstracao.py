from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_ollama import ChatOllama
from langchain.tools.render import render_text_description
from langchain.agents import tool
from langchain.tools import Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad.log import format_log_to_str
from dotenv import load_dotenv
from typing import Union, List, Tuple
from callbacks import AgentCallBackHandler


load_dotenv()


# Link download ollama (default: cpu)
# ollama.com/download

# Link pinecone
# pinecone.io

# Link langsmith
# smith.langchain.com

# Nos imports e objetos, é importante sair dando "." em tudo para conseguir perceber
# todas as possibilidades existentes. Checar atributos, métodos, módulos, etc.


# Quanto maior abstração, normalmente menor a capacidade de customização
@tool
def get_text_length(text: str) -> int:
    "Returns the length of a text by characters"

    print(f"get_text_length enter with {text}")
    stripped_text = text.strip("'\n").strip('"')

    return len(stripped_text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool

    raise ValueError(f"Tool with name '{tool_name} not found.'")


def main():
    tools = [get_text_length]

    # Entender agent_scratchpad é a chave para entender react agents! É basicamente um log de interações da chain!
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOllama(
        model="llama3", stop=["\nObservation"], callbacks=[AgentCallBackHandler()]
    )
    intermediate_steps: List[Tuple[AgentAction, str]] = []

    input_dict = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    react_single_output_parser = ReActSingleInputOutputParser()
    agent = input_dict | prompt | llm | react_single_output_parser

    agent_step = ""  # type: ignore

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length in characters of the text 'DOG'?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)  # type: ignore
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))  # type: ignore
            print()
            print("Observação:", observation)

            intermediate_steps.append((agent_step, str(observation)))
            print(intermediate_steps)

        elif isinstance(agent_step, AgentFinish):
            print(agent_step.return_values)
            print("TENHO A RESPOSTA FINAL!")


if __name__ == "__main__":
    main()
