from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import Dict, List, Tuple, Union
from langchain_core.agents import AgentFinish
from newsletter_gen.tools import Search, FindSimilar, GetContents
import datetime
import json
import os
#from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama




@CrewBase
class NewsletterGenCrew():
    """NewsletterGen crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    

    def llm(self):
        llm = Ollama(model="mixtral-8x7b-32768")
        #llm = ChatGroq(model="llama3-70b-8192")
        #llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", max_tokens=4096)
        #llm = ChatGroq(model="mixtral-8x7b-32768")
        #llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"))
        return llm


    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[Search(), FindSimilar(), GetContents()],
            verbose=True,
            llm=self.llm(),
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config["editor"],
            verbose=True,
            tools=[Search(), FindSimilar(), GetContents()],
            llm=self.llm(),
        )

    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            llm=self.llm(),
            allow_delegation=False,
        )


    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher(),
            output_file=f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_research_task.md',
        )
    
    @task
    def edit_task(self) -> Task:
        return Task(
            config=self.tasks_config["edit_task"],
            agent=self.editor(),
            output_file=f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_edit_task.md',
        )

    @task
    def newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["newsletter_task"],
            agent=self.designer(),
            output_file=f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_newsletter.html',
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NewsletterGen crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,  
        )