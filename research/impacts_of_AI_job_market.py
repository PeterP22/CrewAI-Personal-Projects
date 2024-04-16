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

# Tools
search_tool = SerperDevTool()

topic_agent = Agent(
    llm=llm,
    role="Topic Researcher",
    goal="Identify key information and subtopics related to a research topic.",
    backstory="You are an expert at understanding research topics and finding relevant information.",
    allow_delegation=False,
    tools=[search_tool], 
    verbose=True,
)

# Source Finder Agent
source_agent = Agent(
    llm=llm,
    role="Source Finder",
    goal="Locate credible and relevant sources of information.",
    backstory="You are skilled at finding reliable sources like academic papers, books, and reputable websites.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

# Summarizer Agent
summarizer_agent = Agent(
    llm=llm,
    role="Information Summarizer",
    goal="Condense information from various sources into concise summaries.",
    backstory="You have a knack for extracting key points and creating clear and concise summaries.",
    allow_delegation=False,
    verbose=True,
)

# Research Task
research_task = Task(
    description="Research the topic of 'Artificial Intelligence and its impact on the job market'",
    expected_output="A comprehensive report covering key aspects of AI's impact on the job market, including potential job displacement, new job creation, and required skill adaptations.",
    agent=topic_agent,
    output_file="research_task_output.txt",
)

# Source Finding Task
source_task = Task(
    description="Find 5 credible sources related to the research topic",
    expected_output="A list of 5 sources with titles, authors, and publication information.",
    agent=source_agent,
    input_file="research_task_output.txt",
    output_file="source_task_output.txt",
)

# Summarization Task 
summary_task = Task(
    description="Summarize the key findings from the provided sources",
    expected_output="A concise summary of the main points and insights gathered from the research sources.",
    agent=summarizer_agent,
    input_file="source_task_output.txt",
    output_file="summary_task_output.txt",
)

# Crew Formation and Execution
crew = Crew(agents=[topic_agent, source_agent, summarizer_agent], tasks=[research_task, source_task, summary_task], verbose=2)
task_output = crew.kickoff()
print(task_output)