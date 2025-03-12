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

Output format:
{
    "relations": [
        {
            "head": {"text": "entity_text", "label": "entity_label"},
            "tail": {"text": "entity_text", "label": "entity_label"},
            "relation": "relation_type"
        }
    ]
}
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

EXAMPLE_GLINER_INPUT1 = """
Text: Radio City is India's first private FM radio station and was started on 3 July 2001.

Detected entities:
- Radio City (company) [0:10]
- India (country) [14:19]
- 3 July 2001 (date) [63:74]
"""

EXAMPLE_GLINER_OUTPUT1 = """{
    "relations": [
        {
            "head": {"text": "Radio City", "label": "company"},
            "tail": {"text": "India", "label": "country"},
            "relation": "located_in"
        },
        {
            "head": {"text": "Radio City", "label": "company"},
            "tail": {"text": "3 July 2001", "label": "date"},
            "relation": "established_on"
        }
    ]
}"""

EXAMPLE_GLINER_INPUT2 = """
Text: SpaceX, founded by Elon Musk in 2002, launched its Starlink project from Cape Canaveral.

Detected entities:
- SpaceX (company) [0:6]
- Elon Musk (person) [19:28]
- 2002 (date) [32:36]
- Starlink (project) [51:59]
- Cape Canaveral (location) [65:78]
"""

EXAMPLE_GLINER_OUTPUT2 = """{
    "relations": [
        {
            "head": {"text": "SpaceX", "label": "company"},
            "tail": {"text": "Elon Musk", "label": "person"},
            "relation": "founded_by"
        },
        {
            "head": {"text": "SpaceX", "label": "company"},
            "tail": {"text": "2002", "label": "date"},
            "relation": "founded_on"
        },
        {
            "head": {"text": "SpaceX", "label": "company"},
            "tail": {"text": "Starlink", "label": "project"},
            "relation": "launched"
        },
        {
            "head": {"text": "Starlink", "label": "project"},
            "tail": {"text": "Cape Canaveral", "label": "location"},
            "relation": "launched_from"
        }
    ]
}"""

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
