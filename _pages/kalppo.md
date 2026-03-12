---
layout: post
title: Internship Experience at Kalppo!
permalink: /experience/kalppo/
nav: false
date: 03-12-2025 23:00:00
collection: books
toc:
  sidebar: left
mermaid:
  enabled: true
  zoomable: true
tikzjax: true
typograms: true
---

## Introduction

At the end of my sophomore year at NIT Karnataka, I started searching for research internships in AI/ML at institutes like IITs, IISc, and other research labs. I was very interested in getting hands-on experience in artificial intelligence and machine learning through academic research.

During this process, I was fortunate enough to secure a few research opportunities, including:

- Research Intern at IIT Dharwad  
- Research Intern in Explainable AI at IIT Jammu  

While these were great opportunities, something unexpected happened in April.

## Joining Kalppo
Through the Web Enthusiasts Club (WEC)  a great technical club at NITK I got a referral for an internship at **Kalppo**, an EdTech startup. The referral came from a senior in the club who was already working there as an intern. He introduced me to the team at Kalppo and suggested that I go through the interview process.

I agreed to take the interview.

My technical interview was conducted in the last week of April by [Abhishek Kumar](https://www.linkedin.com/in/abhishekkumar2718/), the CTO and Tech Lead at Kalppo. The interview mostly revolved around my resume, the projects I had worked on, and my experience with machine learning systems.

We discussed things like:

- My work with vector databases  
- The ML systems I had built  
- My experience participating in Kaggle competitions  

One of the highlights discussed during the interview was my result in the **Kaggle × Skill Assessment Machine Learning Competition**, where I was ranked **1st out of 1,846 participants**.

The interview was quite technical but also very interesting because it focused on how I approached problems and built systems, rather than just theoretical questions.

About two days later, I received a call from [Avinash Kumar](https://www.linkedin.com/in/avinash136/) from the HR team at Kalppo. We discussed the role, expectations, and a few details about the internship.

The very next day, I received the confirmation 🙂

My internship at **Kalppo** officially started on **May 10th** and continued until **July 20th**.

Those two months turned out to be one of the most challenging, interesting, and enjoyable experiences I had during my early years in tech. It was my first time working professionally as an AI Engineer Intern, and doing it in a fast-moving EdTech startup environment made the experience even more intense and exciting.

In the following sections, I will talk about the projects I worked on, the people I interacted with, and the things I learned during this journey at Kalppo.


## Team and Mentors

In the first week of May, around 3 interns were selected to join Kalppo. We had our first team meeting where Abhishek Kumar and Avinash Kumar introduced us to the team and the product they had been building over the past six months.

During the meeting, they gave us a demo of the platform and explained the main idea behind the startup. Kalppo is an EdTech startup focused on helping offline coaching institutes bring their teaching systems online.

The product was designed to help coaching institutes:

* create and manage batches of students
* conduct mock quizzes and tests
* manage learning material and progress tracking

The primary focus was on **JEE and NEET coaching**, which makes sense because many coaching institutes in India operate offline on a relatively small scale. The idea was simple but powerful provide them with tools to run their coaching programs online for their students.

Apart from this core product, the team was also working on several other supporting tools and features that could improve the learning experience for both teachers and students.

After the introduction and product demo, we had a fun activity to break the ice. Everyone had to share three statements about themselves, where two statements were true and one was false, and the rest of the group had to guess which one was incorrect.

It was a simple but very engaging activity and helped everyone get comfortable with each other. Since many of us were meeting for the first time, it made the environment much more relaxed.

Overall, it was a great experience meeting the team and mentors. The session helped us understand the **vision behind the startup**, the product they were building, and the people we would be working with over the next few months.


Below is a **clean `.md` section** you can paste under **`## Projects I Worked On`** in your post.
It follows your structure:

1. Brief overview of the three projects
2. Detailed explanation of the **first project (STEM Question Generation)**
3. Includes **Python code blocks, Mermaid diagram, and TikZ example**


## Projects I Worked On

During my internship at Kalppo, I mainly worked on three AI-focused projects that were closely related to the EdTech platform.

STEM Question Generation
The goal of this system was to automatically generate **similar questions for an existing question** so that coaching institutes could expand their **question banks**. This would allow students to practice more variations of a concept instead of repeating the same problems.

AI Question Extraction Pipeline
This pipeline focused on **extracting questions from uploaded documents such as PDFs or scanned worksheets**. The system used OCR and structured parsing to convert raw documents into **structured question objects** that could be stored in the database.

Workbook Evaluation App
This project focused on building an **AI-powered evaluation system** where students could upload workbook answers and the system could automatically evaluate them using LLM-based reasoning and answer matching.


### 1. STEM Question Generation

After our initial onboarding meeting, the following Monday I had a **one-on-one discussion with Abhishek Kumar (CTO)** regarding my experience with AI frameworks and the projects I had previously worked on.

Based on this discussion, the **first major task assigned to me** was to build a system that could generate **k similar questions** for a given input question.

The idea was simple:

- Take an existing JEE-level question
- Generate **multiple variations of the same concept**
- Store them in the database
- Expand the **practice question bank** for coaching institutes

This helps students practice more problems that test the **same concept with slightly different variations**.


#### Experimenting with AI Workflows

I started experimenting with different **AI workflows inside Jupyter notebooks**.

The basic pipeline looked like this:

1. Input question
2. Prompt generation
3. Send prompt to LLM
4. Parse structured JSON output
5. Store in database

#### Prompt Layer

To generate similar questions, I first designed a **prompt layer** responsible for converting input questions into structured prompts for the LLM.

Instead of writing raw prompts everywhere in the codebase, I created a **prompt management layer** that dynamically renders prompts depending on:

- subject (math / physics / chemistry)
- chapter
- number of variations required
- question type

For example, chemistry questions belonging to **organic chemistry chapters** used a different prompt template because they required **SMILES representations for molecular structures**.


#### Prompt Rendering Logic

Below is the simplified Python implementation used to render prompts.

```python
from app.prompts.prompt_manager import PromptManager
from app.schema.question import Question
import json

class GenerateSimilarQuestionPrompt:

    _supported_subjects = {"math", "physics", "chemistry"}

    _organic_chapters = {
        "Alkanes",
        "Alkenes and Alkynes",
        "Alcohols",
        "Aldehydes and Ketones",
        "Carboxylic Acids",
        "Aromatic Compounds (Benzene and its Derivatives)"
    }

    @staticmethod
    def render(questions: list[Question], subject: str, k: int) -> str:

        subject = subject.lower()

        if subject not in GenerateSimilarQuestionPrompt._supported_subjects:
            raise ValueError("Unsupported subject")

        questions_json = json.dumps([
            {
                "id": q.id,
                "question": q.question,
                "subject": q.subject,
                "options": [opt.model_dump() for opt in q.options] if q.options else [],
                "type": q.type,
                "chapter": q.chapter,
            }
            for q in questions
        ], indent=2)

        return PromptManager.render(
            "generate_similar_question_prompt",
            questions_json=questions_json,
            subject=subject,
            k=k
        )
```

This prompt was then sent to the **LLM**, which generated similar questions in **structured JSON format**.


#### Prompt Generation Pipeline

The overall prompt pipeline looked like this:

```mermaid
flowchart LR

A[Input Questions] --> B[Prompt Layer]
B --> C[Prompt Template Rendering]
C --> D[LLM API Call]
D --> E[Structured JSON Output]
E --> F[Pydantic Validation]
F --> G[Stored in Database]
```


#### Why a Prompt Layer Was Important

This abstraction allowed the system to:

* reuse prompt templates across subjects
* dynamically modify prompts depending on chapters
* maintain versioned prompts
* enforce structured outputs

It also made experimentation easier when testing different prompt strategies for **STEM question generation**.


#### LLM Model

For the generation pipeline I used:

* **Gemini 2.5**
* **Gemini 2 Pro**

These models worked well for generating **conceptually similar but diverse questions**.

I used the **Gemini Python API** together with **Pydantic schemas** to validate responses.

#### Question Schema

The generated questions were validated using a **Pydantic schema**.

```mermaid
classDiagram

class GeminiSafeImage {
  +url: str
  +file_name: str
  +mime_type: str
  +base64_bytes: str
}

class Option {
  +label: str
  +text: str
  +images: str
  +is_correct: bool
}

class SimilarQuestion {
  +parent_id: str
  +id: str
  +question: str
  +options: List<Option>
  +correct_answer: str
  +explanation: str
  +type: str
  +source: str
  +difficulty: str
  +skills_tested: List[str>
  +concepts_tested: List[str>
  +subject: str
  +chapter: str
  +images: List<GeminiSafeImage>
}

class GenerateSimilarQuestionResponse {
  +root: List<SimilarQuestion>
}

SimilarQuestion "1" --> "*" Option : contains
SimilarQuestion "1" --> "*" GeminiSafeImage : includes
GenerateSimilarQuestionResponse "1" --> "*" SimilarQuestion : returns
```

This ensured that the LLM output always followed the **correct structured format**.


#### Backend API

Once the pipeline was stable, I built a **FastAPI endpoint** that allowed the system to generate similar questions.

Example endpoint:

```python
from fastapi import APIRouter
from typing import List

router = APIRouter()

@router.post("/generate_similar_question")
def generate_similar_question_route(request):
    similar_questions = generate_similar_question(
        questions=request.questions,
        subject=request.questions[0].subject,
        number_of_generated_questions_per_question=request.number
    )
    return {"similar_questions": similar_questions}
```

The API:

* accepts a list of questions
* generates variations
* returns structured JSON


#### Handling Subject-Specific Content

Each subject required slightly different handling.

#### Mathematics

Math questions mainly relied on **LaTeX formatting** and symbolic expressions.


#### Chemistry (SMILES notation)

For **organic chemistry questions**, I used **SMILES representation** to generate molecular structures.

Example:

```
<smiles>CC(O)=O</smiles>
```

This allowed the system to generate organic chemistry questions involving **molecular structures**.


#### Physics (Circuit Diagrams)

For chapters like:

* Current Electricity
* Alternating Current
* Capacitance

I used **LaTeX + CircuitTikZ** to generate circuit diagrams.

Example:

```latex
\begin{circuitikz}
(0,0) to[sV, l=$V_p\sin(\omega t)$] (0,4)
to[R, l=$R$] (4,4)
to[C, l=$C$] (4,0)
to[short] (0,0);
\end{circuitikz}
```

These diagrams were rendered into images and attached to questions.

<div class="row mt-3">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid 
            loading="eager" 
            path="assets/img/kalppo/tikz.png" 
            class="img-fluid rounded z-depth-1"
        %}
    </div>
</div>

<div class="caption">
Example circuit diagram generated using LaTeX CircuitTikZ and rendered into an image by the pipeline.
</div>


---

#### Example Result: Generated Question Variations

Below is an example showing how the system generates **multiple variations of a given question** while preserving the underlying concept.


#### Original Question

**Question ID:** B-33  

**Question**

Evaluate the integral:

$$
\int_{-1}^{1} \cot^{-1}\left(\frac{x + x^{3}}{1 + x^{4}}\right) \, dx
$$

**Options**

- **A**: $2\pi$
- **B**: $\frac{\pi}{2}$
- **C**: $0$
- **D**: $\pi$

**Correct Answer:** **D** — $\pi$


#### Generated Variation 1

**Question ID:** B-33  
**Sub ID:** B-33-1

**Question**

Evaluate

$$
\int_{0}^{1} \tan^{-1}\left(\frac{2x}{1 - x^2}\right) \, dx
$$

**Options**

- **A**: $\frac{\pi}{2}$
- **B**: $\frac{\pi}{4}$
- **C**: $0$
- **D**: $\pi$

**Correct Answer:** **D** — $\pi$


#### Generated Variation 2

**Question ID:** B-33  
**Sub ID:** B-33-2

**Question**

Find the value of

$$
\int_{-\pi/2}^{\pi/2} \sin^{-1}\left(\frac{2x}{1+x^2}\right) \, dx
$$

**Options**

- **A**: $0$
- **B**: $\pi$
- **C**: $\frac{\pi}{2}$
- **D**: $-\pi$

**Correct Answer:** **A** — $0$


#### System Architecture Diagram

```typograms
-------------------------------------------------------------------------------------------------------.
|                                     STEM QUESTION GENERATION SYSTEM                                  |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                                         FASTAPI BACKEND API                                           |
|                                                                                                       |
|   POST /generate_similar_question                                                                     |
|   - Accepts list of questions                                                                         |
|   - Accepts number_of_generated_questions_per_question                                                |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                                          REQUEST PARSER                                               |
|                                                                                                       |
|  - Pydantic Schema Validation                                                                         |
|  - Question Object Conversion                                                                         |
|  - Subject Detection (Math / Physics / Chemistry)                                                     |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                                            PROMPT LAYER                                               |
|                                                                                                       |
|  Jinja Prompt Template                                                                                |
|                                                                                                       |
|  - GenerateSimilarQuestionPrompt.render()                                                             |
|  - Dynamic Prompt Versioning                                                                          |
|  - Organic Chemistry Detection                                                                        |
|  - Inject Question JSON                                                                               |
|                                                                                                       |
|  Output → Structured LLM Prompt                                                                       |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                                         GEMINI LLM CALL                                               |
|                                                                                                       |
|   Google Gemini API                                                                                   |
|                                                                                                       |
|   Models Used                                                                                         |
|   - gemini-2.5-pro                                                                                    |
|   - gemini-2.0-flash                                                                                  |
|                                                                                                       |
|   Task                                                                                                 |
|   - Generate K Similar Questions                                                                      |
|   - Maintain Concept Consistency                                                                      |
|   - Produce Structured JSON Output                                                                    |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                                      STRUCTURED OUTPUT VALIDATION                                     |
|                                                                                                       |
|  Parse JSON Response                                                                                  |
|                                                                                                       |
|  Pydantic Models                                                                                      |
|     SimilarQuestion                                                                                   |
|     Option                                                                                            |
|     GeminiSafeImage                                                                                   |
|                                                                                                       |
|  Ensures                                                                                              |
|  - Schema correctness                                                                                 |
|  - Correct question structure                                                                         |
|  - Valid options and answers                                                                          |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                               OPTIONAL PHYSICS CIRCUIT DIAGRAM GENERATION                             |
|                                                                                                       |
|   If Chapter ∈ { Current Electricity, Capacitance, AC }                                               |
|                                                                                                       |
|      Gemini → Generate CircuitTikZ LaTeX Code                                                         |
|      ↓                                                                                                |
|      LaTeX Rendering Engine                                                                           |
|      ↓                                                                                                |
|      Convert → PNG Image                                                                              |
|      ↓                                                                                                |
|      Upload → Supabase Storage                                                                        |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                                      DATABASE STORAGE LAYER                                           |
|                                                                                                       |
|  PostgreSQL                                                                                           |
|                                                                                                       |
|  Tables                                                                                                |
|   - Questions                                                                                         |
|   - Options                                                                                           |
|   - Images                                                                                            |
|                                                                                                       |
|  Stored Data                                                                                          |
|   - Generated Question Variations                                                                     |
|   - Answers & Explanations                                                                            |
|   - Circuit Diagrams                                                                                  |
'-------------------------------------------------------------------------------------------------------'
            |
            v
.-------------------------------------------------------------------------------------------------------.
|                                        FINAL API RESPONSE                                             |
|                                                                                                       |
|  FastAPI returns                                                                                      |
|                                                                                                       |
|  GenerateSimilarQuestionResponse                                                                      |
|                                                                                                       |
|  JSON Response                                                                                        |
|                                                                                                       |
|  {                                                                                                    |
|      "similar_questions": [...]                                                                       |
|  }                                                                                                    |
'-------------------------------------------------------------------------------------------------------'
```

### 2. AI Question Extraction Pipeline

One of the major tasks during my internship was building an **AI Question Extraction Pipeline**.

The goal of this system was to automatically **extract questions from educational documents** such as:

- Previous Year Question Papers (PYQs)
- DPPs (Daily Practice Problems)
- Coaching Modules
- Handwritten Notes
- Worksheets
- Mock Test Papers

and convert them into **structured question objects that could be stored in the database**.

Many interns had attempted solving this problem earlier, but building a **reliable production-ready solution** turned out to be quite challenging.


#### Initial Experiments

Initially, I experimented with **DSPy**, a framework designed for building LLM pipelines.

DSPy allows:

- better prompt engineering
- training LLMs on structured datasets
- automatic evaluation of LLM outputs
- optimization of prompts through feedback loops

While DSPy was powerful, the pipeline became **too heavy and complex** for our use case, and the task needed to be delivered quickly.

So I decided to rethink the approach.

#### Microservices Approach

Instead of solving the entire problem at once, I broke the system into **smaller independent services**.

The pipeline was divided into stages:

1. Document → OCR
2. OCR → Markdown text
3. Markdown → Semantic chunks
4. Chunks → Question extraction
5. Validation → Database storage

Breaking the system into microservices allowed each stage to be **independently optimized and debugged**.


#### Step 1: Extracting Text from Documents (OCR)

The first major problem was converting the uploaded **PDF or image documents into textual data**.

For this, I used **Mistral OCR**, which at the time was one of the best OCR systems available.

Key advantages of Mistral OCR:

- Maintains **mathematical formulas in LaTeX**
- Preserves **tables and MCQ formatting**
- Extracts **diagrams and images**
- Maintains **organic chemistry structures**
- Outputs **Markdown formatted text**

This was extremely useful because **JEE / NEET questions heavily depend on mathematical expressions**.


#### Example Extracted Markdown

Below is an example of markdown extracted from a question paper.

```markdown
# SECTION - I  
(SINGLE CORRECT ANSWER TYPE)

This section contains 20 multiple choice questions.

1. Consider the ratio  
$r=\frac{(1-a)}{(1+a)}$  

If the error in measurement of $a$ is $\Delta a$, then the maximum possible error in $r$ is:

1) $\frac{\Delta a}{(1+a)^2}$  
2) $\frac{2\Delta a}{(1+a)^2}$  
3) $\frac{2\Delta a}{(1-a^2)}$  
4) $\frac{2a\Delta a}{(1-a^2)}$
````

As you can see, the OCR output preserves:

* **LaTeX formulas**
* **MCQ options**
* **document structure**

This made it much easier to process questions downstream.

#### OCR Processing Service

Below is a simplified version of the OCR service.

```python
from mistralai import Mistral
from pydantic import BaseModel, HttpUrl
import filetype

client = Mistral(api_key="MISTRAL_API_KEY")

class TranscribeResponse(BaseModel):
    markdown_text: str
    num_pages: int

def transcribe(file_url: HttpUrl) -> TranscribeResponse:

    # call mistral OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": file_url}
    )

    markdown_text = ""

    for page in ocr_response.pages:
        markdown_text += page.markdown + "\n\n"

    return TranscribeResponse(
        markdown_text=markdown_text.strip(),
        num_pages=len(ocr_response.pages)
    )
```

This service:

1. downloads the document
2. sends it to **Mistral OCR**
3. receives structured OCR output
4. converts it into **clean Markdown text**

```mermaid
flowchart LR

A[PDF / Image Upload]
B[Mistral OCR]
C[Markdown Output]
D[Image Extraction]
E[Supabase Storage]
F[Structured Markdown]

A --> B
B --> C
C --> D
D --> E
C --> F
```

At this point, we had successfully converted raw documents into structured Markdown text.

The next challenge was to extract individual questions from this Markdown, which required a completely different AI pipeline.

#### Step 2: Extracting Individual Questions from Markdown

The first step (OCR → Markdown conversion) was already implemented by several interns.

However, the **real challenge** started in the next stage  extracting each individual question in a structured format so that it could be stored in the database.

The OCR output for a typical **JEE test paper containing ~90 questions** produced around:

- **25–30 pages of Markdown**
- roughly **50,000–60,000 characters**

Sending the entire document to an LLM at once was impossible because of token limits.

At that time:

- maximum **output tokens ≈ 32k**
- maximum **input tokens ≈ 8k**

So the document had to be **split into smaller chunks**.

#### The Naive Chunking Approach

The obvious approach would be:

> Split the Markdown text into smaller chunks and send each chunk to the LLM.

However, this approach quickly failed.

Because the chunking was **random**, it often separated:

- the **question statement**
- the **options**
- the **answer / explanation**

into **different chunks**.

Example problem:

```markdown
Chunk 1:
Question statement

Chunk 2:
Options A B C D
```

In such cases, the LLM could not understand that both pieces belonged to the **same question**.

This resulted in:
- orphan questions
- incomplete options
- hallucinated questions

For an EdTech platform, accuracy was extremely important. Coaching institutes wanted **exact question extraction**, not AI-generated approximations.
So this approach was **not acceptable**.

#### Semantic Chunking Attempt

The next idea was to use **semantic chunking**, which is commonly used in industry for splitting large documents before sending them to LLMs.

The idea is simple:

1. Convert text into embeddings
2. Split the document at semantic boundaries

This method is much smarter than random splitting.

Example implementation:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

def generate_semantic_chunks(markdown_text: str):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    chunker = SemanticChunker(
        embeddings,
        min_chunk_size=400,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    return chunker.split_text(markdown_text)
```

While this approach worked well for **general text documents**, it still had problems for **exam papers**.

#### Extraction Pipeline Overview

Or visually:

```mermaid
flowchart LR

A[OCR Markdown Text]
B[Structured Chunking Algorithm]
C[Question-Level Chunks]
D[LLM Extraction]
E[JSON Validation]
F[Database Storage]

A --> B
B --> C
C --> D
D --> E
E --> F
```


#### Step 3: LLM Extraction Layer

Once the document was converted into **structured question-level chunks**, the next step was to extract questions in a **fully structured format**.

Each chunk was sent to an **LLM extraction layer**, where a generalized prompt instructed the model to identify:

- question statements
- options
- correct answers
- explanations
- associated images or diagrams

The goal was to convert raw chunk text into structured **question objects** that could be stored in the database.


#### Parallel Chunk Processing

Since a document could contain **50–100 questions**, each chunk was processed **concurrently** to improve performance.

Below is a simplified version of the extraction pipeline.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def extract_questions_from_chunks(chunks):

    executor = ThreadPoolExecutor(max_workers=4)

    def process_chunk(chunk_text):

        prompt = ExtractQuestionEntityPrompt.render(chunk_text)

        response = gemini.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )

        json_output = parse_json(response.text)

        return ExtractedQuestionEntity.model_validate(json_output)

    loop = asyncio.get_running_loop()

    tasks = [
        loop.run_in_executor(executor, process_chunk, chunk)
        for chunk in chunks
    ]

    results = await asyncio.gather(*tasks)

    entities = []
    for r in results:
        entities.extend(r.root)

    return entities
```

Key ideas in this stage:

* Each chunk is processed **independently**
* Gemini generates **JSON structured outputs**
* Pydantic validates schema correctness
* The system merges entities into final question objects

#### Schema Validation Layer

To ensure the LLM output was **correct and reliable**, I used **Pydantic schemas**.

These schemas validated:

* question text
* options
* answers
* explanations
* associated images

Example simplified schema:

```python
from pydantic import BaseModel
from typing import List, Optional
from enum import StrEnum

class QuestionType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERICAL = "numerical"
    TEXT = "text"

class Option(BaseModel):
    label: str
    text: str
    is_correct: bool

class ExtractedQuestion(BaseModel):
    question_label: str
    question: str
    type: QuestionType
    options: Optional[List[Option]]
    correct_answer: Optional[str]
    explanation: Optional[str]
```
The schemas guaranteed that the LLM output always matched the **expected structure** before storing it in the database.

#### Prompt Rendering Layer

Prompts were rendered dynamically using a **PromptManager**.

```python
class ExtractQuestionEntityPrompt:

    _version = "202506062213"

    @staticmethod
    def render(chunk: str):

        return PromptManager.render(
            "extract_question_entity_prompt",
            ExtractQuestionEntityPrompt._version,
            chunk=chunk
        )
```

This made it easy to **version prompts and improve extraction quality** without changing the code.

#### Question Extraction Pipeline

```mermaid
flowchart LR

A[Markdown Chunk]
B[Prompt Rendering]
C[Gemini API]
D[JSON Output]
E[Pydantic Validation]
F[Merged Question Entities]
G[Database Storage]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
```

#### Extracted Entity Schema

The system supported three entity types:

1. **Question without explanation**
2. **Explanation entity**
3. **Answer key entity**

These entities were later merged into a final **ExtractedQuestion object**.

```mermaid
classDiagram

class ExtractedQuestionWithoutExplanation{
question_label
question
type
options
images
subject
chapter
concepts
}

class ExtractedQuestionExplanation{
question_label
explanation
explanation_images
correct_answer
}

class ExtractedAnswerkey{
question_label
correct_answer
}

class ExtractedQuestionEntity{
List~Entities~
}

ExtractedQuestionEntity --> ExtractedQuestionWithoutExplanation
ExtractedQuestionEntity --> ExtractedQuestionExplanation
ExtractedQuestionEntity --> ExtractedAnswerkey
```

#### Step 4: API Orchestration and Database Storage

After building the OCR layer, chunking layer, and LLM extraction layer, the final step was to expose the entire pipeline through a FastAPI endpoint.

This endpoint acts as the entry point for the extraction system, allowing coaching institutes or internal services to submit a document for question extraction.

The pipeline performs the following steps:

1. Receive extraction request
2. Run OCR on the uploaded document
3. Convert the document into Markdown
4. Split the Markdown into structured chunks
5. Send chunks to the LLM extraction layer
6. Validate structured outputs
7. Store extracted questions in the database

Because extracting questions from a large document could take **several seconds to minutes**, the extraction process was implemented using **background tasks**.

This allowed the API to return immediately while the extraction pipeline continued processing asynchronously.


#### FastAPI Endpoint

Below is a simplified version of the API endpoint.

```python
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()

class ExtractQuestionRequest(BaseModel):
    long_running_operation_id: str

@router.post("/extract_question_entity")
async def extract_question_entity_route(request: ExtractQuestionRequest,
                                         background_tasks: BackgroundTasks):

    background_tasks.add_task(
        process_extract_question_entity_task,
        request.long_running_operation_id
    )

    return {"status": "Processing started"}
```

This endpoint:

* receives a **long running operation id**
* schedules the extraction pipeline
* returns immediately with a **processing status**

#### Extraction Pipeline Orchestration

The background task coordinates all pipeline components.

```python
def process_extract_question_entity_task(operation_id):

    # Step 1: OCR
    transcription = transcribe(file_url)

    # Step 2: Chunking
    if transcription.num_pages < 3:
        chunks = [transcription.markdown_text]
    else:
        chunks = generate_semantic_chunks(transcription.markdown_text)

    # Step 3: LLM Extraction
    extracted_questions = asyncio.run(
        extract_questions_from_chunks(chunks)
    )

    # Step 4: Store results
    save_questions_to_database(extracted_questions)
```

This function combines the **three previously built microservices**:

* OCR service
* chunking service
* LLM extraction service

#### Database Storage

Once questions were extracted and validated, they were stored in the database.

Each question record contained:

* question text
* options
* correct answer
* explanation
* subject
* chapter
* related images

The pipeline stored both:

* **processed questions**
* **raw extraction output**

#### Final Pipeline Architecture

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant OCR as Mistral OCR
    participant Chunker as Chunking Engine
    participant LLM as Gemini LLM
    participant Validator as Pydantic Validation
    participant DB as PostgreSQL
    participant Storage as Supabase Storage

    Client->>FastAPI: Upload document (PDF/Image)
    FastAPI->>OCR: Send file for OCR processing
    OCR-->>FastAPI: Return Markdown text + images

    FastAPI->>Chunker: Generate structured chunks
    Chunker-->>FastAPI: Question-level chunks

    FastAPI->>LLM: Send chunk + extraction prompt
    LLM-->>FastAPI: Structured JSON questions

    FastAPI->>Validator: Validate schema
    Validator-->>FastAPI: Valid ExtractedQuestion objects

    FastAPI->>Storage: Upload extracted images
    Storage-->>FastAPI: Image URLs

    FastAPI->>DB: Store structured questions
    DB-->>FastAPI: Confirmation

    FastAPI-->>Client: Extraction status / results
```


