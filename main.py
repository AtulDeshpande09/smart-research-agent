import os
import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool


# Load API environment keys
load_dotenv()

def load_yaml_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_pipeline():
    # 1. Load Configurations
    agents_config = load_yaml_config('config/agents.yaml')
    tasks_config = load_yaml_config('config/tasks.yaml')
    
    # 2. Ask user for input topic
    print("\nWelcome to the Smart Research Agent Portal")
    user_topic = input("What topic do you want investigated today? ")
    
    # 3. Setup common language model config
    llm = LLM( model="ollama/phi4-mini", base_url="http://localhost:11434")
    
    # 4. Initialize Tools
    search = SerperDevTool()
    scraper = ScrapeWebsiteTool()
    
    # 5. Build Agents from YAML mapping definitions
    researcher_agent = Agent(
        role=agents_config['researcher']['role'],
        goal=agents_config['researcher']['goal'].format(topic=user_topic),
        backstory=agents_config['researcher']['backstory'],
        tools=[search, scraper],
        llm=llm,
        verbose=True
    )
    
    writer_agent = Agent(
        role=agents_config['writer']['role'],
        goal=agents_config['writer']['goal'].format(topic=user_topic),
        backstory=agents_config['writer']['backstory'],
        tools=[], # Writer needs text orchestration skills, no extra web tool packages
        llm=llm,
        verbose=True
    )
    
    # 6. Build Tasks matching agents to definitions
    task_1 = Task(
        description=tasks_config['research_task']['description'].format(topic=user_topic),
        expected_output=tasks_config['research_task']['expected_output'],
        agent=researcher_agent
    )
    
    task_2 = Task(
        description=tasks_config['writing_task']['description'].format(topic=user_topic),
        expected_output=tasks_config['writing_task']['expected_output'],
        agent=writer_agent
    )
    
    # 7. Assemble and Run Crew
    smart_crew = Crew(
        agents=[researcher_agent, writer_agent],
        tasks=[task_1, task_2],
        process=Process.sequential,
        verbose=True
    )
    
    print(f"\nExecuting Multi-Agent workflow for: {user_topic}...\n")
    crew_output = smart_crew.kickoff()
    
    # 8. Save local file result natively
    output_filename = f"report_{user_topic.lower().replace(' ', '_')}.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(str(crew_output))
        
    print(f"\nComplete! Report created cleanly as: '{output_filename}'")

if __name__ == "__main__":
    run_pipeline()