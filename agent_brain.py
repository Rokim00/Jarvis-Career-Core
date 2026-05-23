import os
import asyncio
import litellm
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()

original_completion = litellm.completion
original_acompletion = litellm.acompletion

def strip_cache_breakpoint(kwargs):
    if 'messages' in kwargs:
        for message in kwargs['messages']:
            if isinstance(message, dict):
                message.pop('cache_breakpoint', None)

def patched_completion(*args, **kwargs):
    strip_cache_breakpoint(kwargs)
    return original_completion(*args, **kwargs)

async def patched_acompletion(*args, **kwargs):
    strip_cache_breakpoint(kwargs)
    return await original_acompletion(*args, **kwargs)

litellm.completion = patched_completion
litellm.acompletion = patched_acompletion

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.5
)

my_profile = {
   "name": os.getenv("MASTER_NAME", "Developer"),
   "skills": os.getenv("MASTER_SKILLS", "Python, AI"),
   "projects": os.getenv("MASTER_PROJECTS", "AI Projects")
}

target_job = {
    "platform": "Upwork",
    "title": "Web Developer needed for Responsive Business Website", # Example Web Dev job
    "description": (
        "We are looking for a frontend or full-stack developer to help build a responsive web landing page. "
        "Must be strong in HTML, CSS, and JavaScript. Experience making clean, modern user interfaces is required."
    )
}

strategist_agent = Agent(
    role="Outreach Strategist",
    goal=f"Analyze how {my_profile['name']} perfectly aligns with the targeted job requirements.",
    backstory=(
        "You are an expert technical recruiter. You excel at looking at any technical job post, "
        "breaking down what the client needs, and matching it with the best skills and projects "
        "available in a developer's portfolio."
    ),
    llm=groq_llm,
    verbose=True
)

writer_agent = Agent(
    role="Professional Proposal Writer",
    goal="Draft hyper-personalized, high-converting, non-plagiarized outreach pitches.",
    backstory=(
        "You are a master copywriter specialized in technical freelancing platforms. You write "
        "punchy, highly specific pitches that focus entirely on matching real project execution "
        "to the client's problem."
    ),
    llm=groq_llm,
    verbose=True
)


task_analysis = Task(
    description=(
        f"1. Thoroughly parse this target job listing:\n{target_job['description']}\n\n"
        f"2. Evaluate it against the developer's attributes:\n"
        f"- Languages/Skills: {my_profile['skills']}\n"
        f"- Projects: {my_profile['key_projects']}\n\n"
        "3. Identify the projects or skills from the profile that most directly solve "
        "the client's specific problems. Determine the most logical technical angle to lead with."
    ),
    expected_output="A structured strategic breakdown identifying specific technical leverage points.",
    agent=strategist_agent
)

task_draft_proposal = Task(
    description=(
        "Using the analysis from the Outreach Strategist, draft a hyper-targeted platform proposal.\n\n"
        "STRICT WRITING GUIDELINES:\n"
        "- Do NOT use generic greetings like 'Dear Hiring Manager' or corporate fluff.\n"
        "- Do NOT use over-used AI words like 'delve', 'testament', 'revolutionize', or 'streamline'.\n"
        "- Start the very first sentence directly with how the developer's relevant skills and past "
        "project experience directly address the technical challenges mentioned in the job description.\n"
        "- Keep the total length strictly under 180 words.\n"
        "- Write with a clean, confident, professional developer-to-client tone."
    ),
    expected_output="A hyper-customized, high-impact proposal statement ready for human review.",
    agent=writer_agent,
    human_input=True
)

scout_crew = Crew(
    agents=[strategist_agent, writer_agent],
    tasks=[task_analysis, task_draft_proposal],
    verbose=True,
    cache=False
)

async def run_agent_workflow():
    return await scout_crew.kickoff_async()

if __name__ == "__main__":
    print("\n==========================================================")
    print("🚀 INITIALIZING YOUR UNIVERSAL AGENTIC SCOUT")
    print("==========================================================\n")
    
    final_pitch = asyncio.run(run_agent_workflow())
    
    print("\n==========================================================")
    print("✨ FINAL VERIFIED AND APPROVED PROPOSAL")
    print("==========================================================")
    print(final_pitch)