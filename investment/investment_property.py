import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = "SERPER KEY HERE"

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
search_tool = SerperDevTool()

# Property Research Agent
research_agent = Agent(
    llm=llm,
    role="Property Analyst",
    goal="Analyze real estate trends and identify investment opportunities.",
    backstory="You are a skilled property analyst with expertise in evaluating real estate markets, identifying trends, and pinpointing promising investment opportunities.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

def get_user_input():
    country = input("Enter the country you're interested in: ")
    city = input("Enter the city within that country: ")
    return country, city

def create_expensive_areas_task(country, city):
    description = f"Identify the most expensive neighborhoods or districts in {city}, {country} based on average property prices."
    return Task(
        description=description,
        expected_output="A list of the most expensive areas in the specified city, ranked by average property price.",
        agent=research_agent,
        output_file="expensive_areas.txt"  
    )

def create_investment_areas_task(country, city):
    description = f"Considering factors like affordability, potential for growth, and rental yields, identify promising areas for real estate investment in {city}, {country}."
    return Task(
        description=description,
        expected_output="A list of recommended areas for real estate investment in the specified city, along with justifications for each choice.",
        agent=research_agent,
        output_file="investment_areas.txt"  
    )

# Get user input
country, city = get_user_input()

# Create tasks
expensive_areas_task = create_expensive_areas_task(country, city)
investment_areas_task = create_investment_areas_task(country, city)

# Crew Formation and Execution
crew = Crew(agents=[research_agent], tasks=[expensive_areas_task, investment_areas_task], verbose=2)
task_output = crew.kickoff()
print(task_output)