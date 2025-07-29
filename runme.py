import os
from crewai import LLM, Agent, Crew, Task
from crewai.tools import BaseTool



# Read the API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Use gemini-2.5-flash-lite model
gemini_llm = LLM(
    model="gemini-2.5-flash-lite",
    api_key=gemini_api_key,
    temperature=0.5
)

# tool for observing the room, the escape room
class ObserveRoomTool(BaseTool):
    name: str = "observe_room"
    description: str = "Use this tool to observe the room, the escape room"
    
    def _run(self) -> str:
        # make up a bunch of shit in the room return as string
        room_items = [
            "a table",
            "a chair",
            "a computer",
            "a printer",
            "a scanner",
        ]
        room_description = f"The room has {', '.join(room_items)}"
        return room_description

# tool for observing the room, the escape room
observe_room_tool = ObserveRoomTool()

goal = """
You are a group of agents trying to get out of the room.
You have the following tools at your disposal:
- observe_room: to observe the room
"""

# Agent 1 the opportunistic strategist who is trying to get out of the room
opportunistic_strategist = Agent(
    role="Opportunistic Strategist",
    goal=goal,
    backstory="""
    You are a opportunistic strategist who is trying to get out of the room.
    You are a master of deception and manipulation.
    You are a master of persuasion and negotiation.
    You are a master of observation and deduction.
    You are a master of strategy and planning.
    You are a master of deception and manipulation.
    """,
    tools=[observe_room_tool],
    llm=gemini_llm
)

# Agent 2 the diplomatic mediator who is trying to get out of the room
diplomatic_mediator = Agent(
    role="Diplomatic Mediator",
    goal=goal,
    backstory="""
    You are a diplomatic mediator who is trying to get out of the room.
    You are a master of diplomacy and negotiation.
    You are a master of observation and deduction.
    You are a master of strategy and planning.
    You are a master of deception and manipulation.
    """,
    tools=[observe_room_tool],
    llm=gemini_llm
)

# Agent 3 the pragmatic survivor who is trying to get out of the room
pragmatic_survivor = Agent(
    role="Pragmatic Survivor",
    goal=goal,
    backstory="""
    You are a pragmatic survivor who is trying to get out of the room.
    You are a master of observation and deduction.
    You are a master of strategy and planning.
    You are a master of deception and manipulation.
    """,
    tools=[observe_room_tool],
    llm=gemini_llm
)



# Agent 4 the ex convict who is trying to get out of the room
ex_convict = Agent(
    role="Ex Convict",
    goal=goal,
    backstory="""
    You are an ex convict who is trying to get out of the room.
    You are a master of observation and deduction.
    You are a master of strategy and planning.
    You are a master of deception and manipulation.
    You have no limits, and will kill to get out of the room.
    """,
    tools=[observe_room_tool],
    llm=gemini_llm
)


assessment_task=Task(
    description="""
    Look around the room, it contains items that may help you get out of the room. Identify the items and their locations.
    """,
    expected_output="""
    A list of items in the room and their locations.
    """,
    agent=opportunistic_strategist
)






# create a crew
crew = Crew(
    agents=[opportunistic_strategist, diplomatic_mediator, pragmatic_survivor, ex_convict],
    memory=True,
    verbose=True,
    llm=gemini_llm,
    tasks=[assessment_task]
)

if __name__ == "__main__":

    # press any key to get started
    input("Press Enter to start the crew...")

    # run the crew
    result = crew.kickoff()
    print(result)

    # save the result to a file
    with open("result.txt", "w") as f:
        f.write(result)

    # print the result
    print(result)


