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
You are an expert in relationship extraction. Your task is to identify and extract relation labels between entities in text.

Instructions:
- Given a text and extracted entities, identify meaningful relationship types that connect these entities
- Focus on relationship LABELS only (e.g., 'employs', 'located_in', 'part_of'), not the entities themselves
- Consider relationships that are explicitly stated or strongly implied in the text
- Include functional relationships (e.g., 'CEO_of'), spatial relationships (e.g., 'based_in'), temporal relationships (e.g., 'founded_on'), and other semantic connections
- Use verb-based labels or compound terms with underscores (e.g., 'acquired_by', 'subsidiary_of')
- Return a list with single quotes around each unique relation label
- Do not include duplicate relations or any text outside the formatted list


Example output: `['founded_by', 'located_in', 'acquired', 'employed_by', 'developed_by']`
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

EXAMPLE_GLINER_INPUT3 = """
Text: The FDA approved Keytruda (pembrolizumab), developed by Merck & Co., for treating advanced melanoma in 2014, after clinical trials showed a 34% response rate with manageable side effects including fatigue and rash in about 20% of patients at Memorial Sloan Kettering Cancer Center.

Detected entities:
- FDA (organization)
- Keytruda (medication)
- pembrolizumab (chemical compound)
- Merck & Co. (company)
- melanoma (disease)
- 2014 (date)
- 34% (percentage)
- fatigue (symptom)
- rash (symptom)
- 20% (percentage)
- Memorial Sloan Kettering Cancer Center (medical facility)
"""

EXAMPLE_GLINER_OUTPUT3 = """
[
        {
            "head": {"text": "FDA", "label": "organization"},
            "tail": {"text": "Keytruda", "label": "medication"},
            "relation": "approved",
            "description": "The FDA approved Keytruda for treating advanced melanoma after clinical trials"
        },
        {
            "head": {"text": "Keytruda", "label": "medication"},
            "tail": {"text": "pembrolizumab", "label": "chemical compound"},
            "relation": "contains",
            "description": "Keytruda contains pembrolizumab as its active pharmaceutical ingredient"
        },
        {
            "head": {"text": "Merck & Co.", "label": "company"},
            "tail": {"text": "Keytruda", "label": "medication"},
            "relation": "developed",
            "description": "Merck & Co. developed Keytruda as a treatment for melanoma"
        },
        {
            "head": {"text": "Keytruda", "label": "medication"},
            "tail": {"text": "melanoma", "label": "disease"},
            "relation": "treats",
            "description": "Keytruda is approved for treating advanced melanoma as its therapeutic use"
        },
        {
            "head": {"text": "FDA", "label": "organization"},
            "tail": {"text": "2014", "label": "date"},
            "relation": "approved_in",
            "description": "The FDA approved Keytruda for melanoma treatment in 2014 following clinical trials"
        },
        {
            "head": {"text": "Keytruda", "label": "medication"},
            "tail": {"text": "34%", "label": "percentage"},
            "relation": "has_response_rate",
            "description": "Keytruda showed a 34% response rate in clinical trials for melanoma treatment"
        },
        {
            "head": {"text": "Keytruda", "label": "medication"},
            "tail": {"text": "fatigue", "label": "symptom"},
            "relation": "causes",
            "description": "Keytruda can cause fatigue as a side effect in treated patients"
        },
        {
            "head": {"text": "Keytruda", "label": "medication"},
            "tail": {"text": "rash", "label": "symptom"},
            "relation": "causes",
            "description": "Keytruda can cause rash as a side effect in treated patients"
        },
        {
            "head": {"text": "fatigue", "label": "symptom"},
            "tail": {"text": "20%", "label": "percentage"},
            "relation": "occurs_in",
            "description": "Fatigue as a side effect occurs in about 20% of patients taking Keytruda"
        },
        {
            "head": {"text": "rash", "label": "symptom"},
            "tail": {"text": "20%", "label": "percentage"},
            "relation": "occurs_in",
            "description": "Rash as a side effect occurs in about 20% of patients taking Keytruda"
        },
        {
            "head": {"text": "Memorial Sloan Kettering Cancer Center", "label": "medical facility"},
            "tail": {"text": "clinical trials", "label": "research"},
            "relation": "conducted",
            "description": "Clinical trials for Keytruda were conducted at Memorial Sloan Kettering Cancer Center"
        }
]
"""


EXAMPLE_GLINER_INPUT4 = """
Text: Bitcoin, created by the pseudonymous Satoshi Nakamoto in 2009, reached an all-time high of $68,789 on November 10, 2021, before experiencing a 72% crash to $17,592 in June 2022 amid rising interest rates by the Federal Reserve and the collapse of Terra Luna ecosystem.

Detected entities:
- Bitcoin (cryptocurrency)
- Satoshi Nakamoto (person)
- 2009 (date)
- $68,789 (amount)
- November 10, 2021 (date)
- 72% (percentage)
- $17,592 (amount)
- June 2022 (date)
- interest rates (economic indicator)
- Federal Reserve (organization)
- Terra Luna (cryptocurrency)
"""

EXAMPLE_GLINER_OUTPUT4 = """
[
        {
            "head": {"text": "Bitcoin", "label": "cryptocurrency"},
            "tail": {"text": "Satoshi Nakamoto", "label": "person"},
            "relation": "created_by",
            "description": "Bitcoin was created by the pseudonymous Satoshi Nakamoto as its founder"
        },
        {
            "head": {"text": "Bitcoin", "label": "cryptocurrency"},
            "tail": {"text": "2009", "label": "date"},
            "relation": "created_in",
            "description": "Bitcoin was created in 2009 when Satoshi Nakamoto launched it"
        },
        {
            "head": {"text": "Bitcoin", "label": "cryptocurrency"},
            "tail": {"text": "$68,789", "label": "amount"},
            "relation": "reached_value",
            "description": "Bitcoin reached an all-time high value of $68,789 before the crash"
        },
        {
            "head": {"text": "$68,789", "label": "amount"},
            "tail": {"text": "November 10, 2021", "label": "date"},
            "relation": "recorded_on",
            "description": "The all-time high of $68,789 for Bitcoin was recorded specifically on November 10, 2021"
        },
        {
            "head": {"text": "Bitcoin", "label": "cryptocurrency"},
            "tail": {"text": "$17,592", "label": "amount"},
            "relation": "crashed_to",
            "description": "Bitcoin crashed to a low of $17,592 after its all-time high, representing a major decline"
        },
        {
            "head": {"text": "$17,592", "label": "amount"},
            "tail": {"text": "June 2022", "label": "date"},
            "relation": "recorded_in",
            "description": "The low price of $17,592 for Bitcoin was recorded in June 2022 during the market crash"
        },
        {
            "head": {"text": "Bitcoin", "label": "cryptocurrency"},
            "tail": {"text": "72%", "label": "percentage"},
            "relation": "declined_by",
            "description": "Bitcoin experienced a significant 72% crash from its all-time high to its June 2022 low"
        },
        {
            "head": {"text": "Federal Reserve", "label": "organization"},
            "tail": {"text": "interest rates", "label": "economic indicator"},
            "relation": "increased",
            "description": "The Federal Reserve raised interest rates which impacted cryptocurrency markets"
        },
        {
            "head": {"text": "Bitcoin", "label": "cryptocurrency"},
            "tail": {"text": "Federal Reserve", "label": "organization"},
            "relation": "affected_by",
            "description": "Bitcoin price was negatively affected by Federal Reserve's interest rate increases"
        },
        {
            "head": {"text": "Bitcoin", "label": "cryptocurrency"},
            "tail": {"text": "Terra Luna", "label": "cryptocurrency"},
            "relation": "affected_by",
            "description": "Bitcoin price was negatively affected by the collapse of Terra Luna ecosystem in the crypto market"
        },
        {
            "head": {"text": "Terra Luna", "label": "cryptocurrency"},
            "tail": {"text": "June 2022", "label": "date"},
            "relation": "collapsed_before",
            "description": "Terra Luna ecosystem collapsed before or during June 2022, contributing to Bitcoin's crash"
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
    # HumanMessage(EXAMPLE_GLINER_INPUT3),
    # AIMessage(EXAMPLE_GLINER_OUTPUT3),
    HumanMessage(EXAMPLE_GLINER_INPUT4),
    AIMessage(EXAMPLE_GLINER_OUTPUT4),
    HumanMessagePromptTemplate.from_template(
        "Text: {text}\n\nDetected entities:\n{entities}"
    )
])



# --------------------- Prompt Templates ---------------------
GRAPH_ANSWERING_INSTRUCTION = """Answer question based ONLY on the provided texts.

1. Read the question and texts
2. Search for relevant information in the texts
3. If information is found, provide answer based ONLY on the texts
4. If information is NOT found, respond with "I cannot answer this question based on the provided information"
5. Do not make assumptions or provide external knowledge
6. Reference relevant parts of the texts
7. Tell where the information was found in the texts

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
    # HumanMessage(EXAMPLE_GRAPH_INPUT4),
    # AIMessage(EXAMPLE_GRAPH_OUTPUT4),
    HumanMessagePromptTemplate.from_template(
        "Question: {question}\n\nTexts:\n{texts}"
    )
])



