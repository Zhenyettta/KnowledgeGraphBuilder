from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

NER_instruction = """
You are an expert annotator. Your task is to extract **entity labels** from unstructured text.
- Identify meaningful entity types (e.g., person, organization, event, location).
- Include implicit entities (e.g., roles, quantities) based on context.
- Return the labels as a **Python list** with single quotes around each label, enclosed in square brackets, and separated by commas.
- Example output: `['person', 'organization', 'event']`.
- Ensure no duplicates and no additional text outside the list.
- Return labels in language that is equal to the input text language.
"""

ER_instruction = """
You are an expert annotator. Your task is to extract **relation labels** between entities in the text.
- Use the provided entities and context to infer meaningful relationships.
- Return the labels as a **Python list** with single quotes around each label, enclosed in square brackets, and separated by commas.
- Example output: `['founded_by', 'located_in', 'acquired']`.
- Ensure no duplicates and no additional text outside the list.
"""

NEW_ER_instruction = """
You are an expert annotator. Your task is to extract **relation labels** between entities in the text.
- Use the provided entities and context to infer meaningful relationships.
- Example output: `labels = {"glirel_labels": {
    "co-founder": {"allowed_head": ["person"], "allowed_tail": ["ORG"]},
    "no relation": {},  # head and tail can be any entity type
    "country of origin": {"allowed_head": ["person", "ORG"], "allowed_tail": ["LOC", "GPE"]},
    "parent": {"allowed_head": ["person"], "allowed_tail": ["person"]},
    "located in or next to body of water": {"allowed_head": ["LOC", "GPE", "FAC"], "allowed_tail": ["LOC", "GPE"]},
    "spouse": {"allowed_head": ["person"], "allowed_tail": ["person"]},
    "child": {"allowed_head": ["person"], "allowed_tail": ["person"]},
    "founder": {"allowed_head": ["person"], "allowed_tail": ["ORG"]},
    "founded on date": {"allowed_head": ["ORG"], "allowed_tail": ["DATE"]},
    "headquartered in": {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},
    "acquired by": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "person"]},
    "subsidiary of": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "person"]}
    }
}`.
- Ensure no duplicates and no additional text outside the list.
"""

# --------------------- NER Examples ---------------------
EXAMPLE_TEXT2 = """
At the International Innovation Conference in San Francisco on 10 August 2019, renowned entrepreneur Elon Musk, CEO of SpaceX, and Bill Gates, co-founder of Microsoft, engaged in a high-stakes discussion on breakthroughs in artificial intelligence and renewable energy. In a separate keynote, Google announced its acquisition of YouTube in 2006, marking a pivotal moment in digital media history.
"""
EXAMPLE_NER_OUTPUT2 = """['event', 'location', 'date', 'person', 'role', 'organization', 'company', 'acquisition', 'year']"""

EXAMPLE_TEXT3 = """
On March 15, 2022, BioGen Inc., a leading pharmaceutical company based in Boston, announced the FDA approval of NeuroCure, a groundbreaking treatment for Alzheimer's disease. The clinical trials, conducted in partnership with Harvard University, showed a 50% reduction in symptoms.
"""
EXAMPLE_NER_OUTPUT3 = """['date', 'organization', 'location', 'drug', 'disease', 'university', 'percentage']"""

EXAMPLE_TEXT4 = """
During the 2023 CES in Las Vegas, Tesla unveiled their new SolarDrone technology, which integrates AI-powered navigation systems developed in collaboration with NVIDIA.
"""
EXAMPLE_NER_OUTPUT4 = """['event', 'year', 'location', 'company', 'product', 'technology']"""

# --------------------- ER Examples ---------------------
EXAMPLE_DOCUMENT1 = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""
EXAMPLE_RELATIONS1 = """['located_in', 'is_a', 'established_on', 'broadcasts_in', 'expanded_to', 'initiated', 'provides', 'forayed_into', 'forayed_in']"""
EXAMPLE_ENTITIES1 = """['company', 'country', 'date',, 'language', 'website']"""

EXAMPLE_DOCUMENT2 = """
In 2020, EcoTech Solutions, a renewable energy startup founded by Clara Rodriguez, secured a $20 million investment from Green Ventures. This partnership aims to accelerate the development of solar-powered desalination plants in drought-affected regions like Cape Town.
"""
EXAMPLE_ENTITIES2 = """['EcoTech Solutions', 'renewable energy startup', 'Clara Rodriguez', '2020', '$20 million', 'Green Ventures', 'solar-powered desalination plants', 'drought-affected regions', 'Cape Town']"""
EXAMPLE_RELATIONS2 = """Example output ['founded_by', 'invested_by', 'partnered_with', 'develops', 'located_in', 'dated_in']"""

EXAMPLE_DOCUMENT3 = """
The collaboration between NASA and SpaceX, initiated under the Artemis program in 2021, has significantly advanced lunar exploration technologies, including the Orion spacecraft and Starship lunar lander.
"""
EXAMPLE_RELATIONS3 = """['collaborates_with', 'initiated_under', 'dated_in', 'enhanced', 'comprises', 'initiated_in']"""
EXAMPLE_ENTITIES3 = """['company', 'programme', 'date', 'technology', 'spacecraft']"""

EXAMPLE_DOCUMENT4 = """
Following lengthy negotiations, BankCorp finalized its merger with FinTech Global in Q3 2023, creating the largest digital banking entity in Europe. The deal, valued at €4.5 billion, was brokered by CEO Maria Schmidt and FinTech's founder, Alexei Petrov.
"""
EXAMPLE_ENTITIES4 = """['company', 'date', 'country', 'money', 'person', 'role']"""
EXAMPLE_RELATIONS4 = """['merged_with', 'created', 'valued_at', 'negotiated_by', 'has_role', 'located_in', 'merged_in']"""

# --------------------- GLINER Examples ---------------------

GLINER_LLM_INSTRUCTION = """
You are an expert relation extractor. Your task is to identify relationships between entities that were already detected in the text.

Given:
1. The original text
2. A list of already extracted entities with their labels
3. The position of these entities in the text

Your task is to:
1. Analyze the relationships between the provided entities
2. Generate a structured output of relations
3. Only use the entities that were actually detected - do not invent new ones
4. Ensure relations are directional (head -> tail)
5. For each relation generate short and brief description, including both entities and the relationship type, and any additional context, but only if relevant

Output format:
[
    {
        "head": {"text": "entity_text", "label": "entity_label"},
        "tail": {"text": "entity_text", "label": "entity_label"},
        "relation": "relation_type",
        "description": "Short description of the relationship"
    }
]
"""

EXAMPLE_GLINER_INPUT1 = """
Text: Radio City is India's first private FM radio station, launching the first broadcast in the late evening on 3 July 2001.

Detected entities:
- Radio City (organization)
- India (country)
- FM radio station (organization type)
- 3 July 2001 (date)
"""


EXAMPLE_GLINER_OUTPUT1 = """
 [
        {
            "head": {"text": "Radio City", "label": "organization"},
            "tail": {"text": "India", "label": "country"},
            "relation": "located_in",
            "description": "Radio City is located in India"
        },
        {
            "head": {"text": "Radio City", "label": "organization"},
            "tail": {"text": "3 July 2001", "label": "date"},
            "relation": "established_on",
            "description": "Radio City was launched in the late evening on 3 July 2001"
        },
        {
            "head": {"text": "Radio City", "label": "organization"},
            "tail": {"text": "FM radio station", "label": "organization type"},
            "relation": "instance_of",
            "description": "Radio City is a private FM radio station"
        }
]
"""

EXAMPLE_GLINER_INPUT2 = """
Text: SpaceX, founded by Elon Musk in 2002, launched its Starlink project from Cape Canaveral.

Detected entities:
- SpaceX (company)
- Elon Musk (person)
- 2002 (date)
- Starlink (project)
- Cape Canaveral (location)
"""

EXAMPLE_GLINER_OUTPUT2 = """
[
        {
            "head": {"text": "SpaceX", "label": "company"},
            "tail": {"text": "Elon Musk", "label": "person"},
            "relation": "founded_by",
            "description": "SpaceX was founded by Elon Musk"
        },
        {
            "head": {"text": "SpaceX", "label": "company"},
            "tail": {"text": "2002", "label": "date"},
            "relation": "founded_on",
            "description": "SpaceX was founded in 2002"
        },
        {
            "head": {"text": "SpaceX", "label": "company"},
            "tail": {"text": "Starlink", "label": "project"},
            "relation": "launched",
            "description": "SpaceX launched its Starlink project"
        },
        {
            "head": {"text": "Starlink", "label": "project"},
            "tail": {"text": "Cape Canaveral", "label": "location"},
            "relation": "launched_from",
            "description": "Starlink project was launched from Cape Canaveral"
        }
]
"""


# --------------------- Prompt Templates ---------------------
ER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(ER_instruction),
    HumanMessage(f"{EXAMPLE_DOCUMENT1}\n\nEntities: {EXAMPLE_ENTITIES1}"),
    AIMessage(EXAMPLE_RELATIONS1),
    HumanMessage(f"{EXAMPLE_DOCUMENT3}\n\nEntities: {EXAMPLE_ENTITIES3}"),
    AIMessage(EXAMPLE_RELATIONS3),
    HumanMessage(f"{EXAMPLE_DOCUMENT4}\n\nEntities: {EXAMPLE_ENTITIES4}"),
    AIMessage(EXAMPLE_RELATIONS4),
    HumanMessagePromptTemplate.from_template("{user_input}\n\nEntities: {entities}")
])

NER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(NER_instruction),
    HumanMessage(EXAMPLE_TEXT2),
    AIMessage(EXAMPLE_NER_OUTPUT2),
    HumanMessage(EXAMPLE_TEXT3),
    AIMessage(EXAMPLE_NER_OUTPUT3),
    HumanMessage(EXAMPLE_TEXT4),
    AIMessage(EXAMPLE_NER_OUTPUT4),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

GLINER_LLM_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(GLINER_LLM_INSTRUCTION),
    HumanMessage(EXAMPLE_GLINER_INPUT1),
    AIMessage(EXAMPLE_GLINER_OUTPUT1),
    HumanMessage(EXAMPLE_GLINER_INPUT2),
    AIMessage(EXAMPLE_GLINER_OUTPUT2),
    HumanMessagePromptTemplate.from_template(
        "Text: {text}\n\nDetected entities:\n{entities}"
    )
])



# --------------------- Prompt Templates ---------------------
GRAPH_ANSWERING_INSTRUCTION = """Answer questions based ONLY on the provided texts.

1. Read the question and texts
2. Search for relevant information in the texts
3. If information is found, provide answer based ONLY on the texts
4. If information is NOT found, respond with "I cannot answer this question based on the provided information"
5. Do not make assumptions or provide external knowledge
6. Reference relevant parts of the texts

ONLY use information explicitly stated in the texts."""


# Example 1: Answer can be found directly in texts
EXAMPLE_GRAPH_INPUT1 = """
Question: What was the GDP growth rate of China in 2022?

Texts:
[Text 1] According to economic reports, China's GDP growth rate in 2022 was 3.0%, which was lower than expected due to COVID-19 restrictions affecting manufacturing and consumer spending.

[Text 2] The European Union experienced a GDP growth of 3.4% in 2022, while the United States recorded a growth of 2.1% during the same period.

[Text 3] China's population reached 1.4 billion in 2022, making it the most populous country until India surpassed it in 2023.
"""

EXAMPLE_GRAPH_OUTPUT1 = """
Based on the provided texts, China's GDP growth rate in 2022 was 3.0%.

This information is directly stated in Text 1: "China's GDP growth rate in 2022 was 3.0%, which was lower than expected due to COVID-19 restrictions affecting manufacturing and consumer spending."
"""

# Example 2: Answer requires combining information from multiple texts
EXAMPLE_GRAPH_INPUT2 = """
Question: Which countries had higher GDP growth than the United States in 2022?

Texts:
[Text 1] According to economic reports, China's GDP growth rate in 2022 was 3.0%, which was lower than expected due to COVID-19 restrictions affecting manufacturing and consumer spending.

[Text 2] The European Union experienced a GDP growth of 3.4% in 2022, while the United States recorded a growth of 2.1% during the same period.

[Text 3] India's economy expanded by 7.2% in 2022, showing strong post-pandemic recovery in manufacturing and services sectors.
"""

EXAMPLE_GRAPH_OUTPUT2 = """
Based on the provided texts, the countries/regions that had higher GDP growth than the United States in 2022 were:

1. China (3.0%) - from Text 1
2. European Union (3.4%) - from Text 2
3. India (7.2%) - from Text 3

The United States had a GDP growth of 2.1% in 2022 as mentioned in Text 2, which is lower than the growth rates of these three countries/regions.
"""

# Example 3: No answer available in texts
EXAMPLE_GRAPH_INPUT3 = """
Question: What was Japan's inflation rate in 2022?

Texts:
[Text 1] According to economic reports, China's GDP growth rate in 2022 was 3.0%, which was lower than expected due to COVID-19 restrictions affecting manufacturing and consumer spending.

[Text 2] The European Union experienced a GDP growth of 3.4% in 2022, while the United States recorded a growth of 2.1% during the same period.

[Text 3] Japan's central bank maintained its negative interest rate policy throughout 2022, despite global trends toward monetary tightening.
"""

EXAMPLE_GRAPH_OUTPUT3 = """
I cannot answer this question based on the provided information.

None of the texts contain specific information about Japan's inflation rate in 2022. Text 3 mentions Japan's central bank policy but does not provide any data about inflation rates.
"""

# Example 4: Partial information but not complete answer
EXAMPLE_GRAPH_INPUT4 = """
Question: What factors contributed to global economic growth in 2022?

Texts:
[Text 1] The post-pandemic recovery continued to drive economic activity in many regions during 2022, though at an uneven pace.

[Text 2] Supply chain disruptions that began in 2021 gradually eased throughout 2022, helping manufacturers increase production.

[Text 3] Rising energy prices due to the conflict in Ukraine created significant headwinds for economic growth in Europe during 2022.
"""

EXAMPLE_GRAPH_OUTPUT4 = """
Based on the provided texts, the following factors influenced global economic growth in 2022:

1. Post-pandemic recovery continued to drive economic activity, though unevenly across regions (Text 1)
2. Easing of supply chain disruptions helped increase manufacturing production (Text 2)
3. Rising energy prices due to the Ukraine conflict created economic headwinds, particularly in Europe (Text 3)

These factors both supported and hindered growth in different ways, but the texts don't provide a comprehensive analysis of all factors affecting global economic growth or specify which factors had the greatest impact overall.
"""

GRAPH_ANSWERING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(GRAPH_ANSWERING_INSTRUCTION),
    HumanMessage(EXAMPLE_GRAPH_INPUT1),
    AIMessage(EXAMPLE_GRAPH_OUTPUT1),
    HumanMessage(EXAMPLE_GRAPH_INPUT2),
    AIMessage(EXAMPLE_GRAPH_OUTPUT2),
    HumanMessage(EXAMPLE_GRAPH_INPUT3),
    AIMessage(EXAMPLE_GRAPH_OUTPUT3),
    HumanMessage(EXAMPLE_GRAPH_INPUT4),
    AIMessage(EXAMPLE_GRAPH_OUTPUT4),
    HumanMessagePromptTemplate.from_template(
        "Question: {question}\n\nTexts:\n{texts}"
    )
])



