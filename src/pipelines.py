from src.deepresearch_agent import research_topic
from src.planner_agent import plan_topics


def process_question(user_question: str) -> str:
    # Get the list of topics from planner agent
    topics = plan_topics(user_question)
    
    # Research each topic using deepresearch agent
    detailed_answers = []
    sources = []

    for topic in topics:
        print(f'Searching the internet for information on the topic: {topic}')
        topic_research, new_sources = research_topic(topic, user_question)
        detailed_answers.append(f"## {topic}\n\n{topic_research}")
        sources.extend(new_sources)
    
    # Combine all research into one comprehensive answer
    final_answer = (
        f"# Answer to: {user_question}\n\n" + "\n\n".join(detailed_answers)
    )

    filename = f'deepresearch.md'
    with open(filename, 'w') as f:
        f.write(final_answer + "\n\nSources:\n\n" + "\n\n".join(sources))
        print(f'Research report saved to {filename}')

    return final_answer
