# CHECKPOINT 1 — "Hello, AI builder"
# This is the file every attendee starts with. If you can see this page
# in the preview on the right, your setup is done.
# It uses the same page layout as the final app, so the look never changes.

import json

import streamlit as st
from google import genai
from google.genai import errors, types

# Prompt for Gemini
SYSTEM_PROMPT = """
You are a friendly and honest career counsellor for students and job seekers in India.
Create a realistic beginner-friendly career plan with exactly 3 roles, 4 to 6 skill gaps,
and a 90-day roadmap split into Days 1-30, Days 31-60, and Days 61-90, with 3 tasks
and one mini project in each period. Suggest one action that can be completed in under
30 minutes today. Follow the provided JSON schema exactly.
Rules: Never promise a job or a fixed salary. Label every salary as indicative. Use simple English.
Keep the answer under 450 words and give advice that is realistic for a beginner.
"""

ROADMAP_PERIOD_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tasks": {
            "type": "ARRAY",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "STRING"},
        },
        "mini_project": {"type": "STRING"},
    },
    "required": ["tasks", "mini_project"],
}

CAREER_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "roles": {
            "type": "ARRAY",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "why_it_fits": {"type": "STRING"},
                    "match_score": {"type": "INTEGER", "minimum": 0, "maximum": 100},
                    "salary_range_lpa": {"type": "STRING"},
                },
                "required": ["title", "why_it_fits", "match_score", "salary_range_lpa"],
            },
        },
        "skill_gaps": {
            "type": "ARRAY",
            "minItems": 4,
            "maxItems": 6,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "skill": {"type": "STRING"},
                    "current_level": {"type": "INTEGER", "minimum": 0, "maximum": 10},
                    "needed_level": {"type": "INTEGER", "minimum": 0, "maximum": 10},
                    "free_resource": {"type": "STRING"},
                },
                "required": ["skill", "current_level", "needed_level", "free_resource"],
            },
        },
        "roadmap": {
            "type": "OBJECT",
            "properties": {
                "days_1_30": ROADMAP_PERIOD_SCHEMA,
                "days_31_60": ROADMAP_PERIOD_SCHEMA,
                "days_61_90": ROADMAP_PERIOD_SCHEMA,
            },
            "required": ["days_1_30", "days_31_60", "days_61_90"],
        },
        "todays_action": {"type": "STRING"},
    },
    "required": ["roles", "skill_gaps", "roadmap", "todays_action"],
}

# Schema for the interview question generator.
INTERVIEW_QUESTIONS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "difficulty": {
                        "type": "STRING",
                        "enum": ["Simple", "Intermediate"],
                    },
                },
                "required": ["question", "difficulty"],
            },
        }
    },
    "required": ["questions"],
}


def create_gemini_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(api_version="v1beta"),
    )


def generate_text(client, contents, system_instruction, response_schema=None):
    last_error = None
    for model, attempts in [("gemini-3.5-flash-lite", 2)]:
        for _ in range(attempts):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    response_mime_type="application/json" if response_schema else None,
                    response_schema=response_schema,
                )
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                if not response.text:
                    raise ValueError("Gemini returned an empty answer.")
                return response.text
            except errors.APIError as exc:
                if exc.code not in (429, 500, 503, 504):
                    raise
                last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("Gemini returned no answer.")


def parse_career_plan(answer):
    plan = json.loads(answer)
    if not isinstance(plan, dict):
        raise ValueError("The plan must be a JSON object.")
    if not isinstance(plan.get("roles"), list) or len(plan["roles"]) != 3:
        raise ValueError("The plan must include exactly three roles.")
    for role in plan["roles"]:
        if not isinstance(role, dict) or not all(
            isinstance(role.get(field), str)
            for field in ("title", "why_it_fits", "salary_range_lpa")
        ):
            raise ValueError("A role is missing required text.")
        if type(role.get("match_score")) is not int or not 0 <= role["match_score"] <= 100:
            raise ValueError("A role has an invalid match score.")
    if not isinstance(plan.get("skill_gaps"), list) or not 4 <= len(plan["skill_gaps"]) <= 6:
        raise ValueError("The plan must include four to six skill gaps.")
    for gap in plan["skill_gaps"]:
        if not isinstance(gap, dict) or not all(
            isinstance(gap.get(field), str) for field in ("skill", "free_resource")
        ):
            raise ValueError("A skill gap is missing required text.")
        if any(
            type(gap.get(field)) is not int or not 0 <= gap[field] <= 10
            for field in ("current_level", "needed_level")
        ):
            raise ValueError("A skill gap has an invalid level.")
    roadmap = plan.get("roadmap")
    if not isinstance(roadmap, dict):
        raise ValueError("The plan is missing its roadmap.")
    for period in ("days_1_30", "days_31_60", "days_61_90"):
        section = roadmap.get(period)
        if (
            not isinstance(section, dict)
            or not isinstance(section.get("tasks"), list)
            or len(section["tasks"]) != 3
            or not all(isinstance(task, str) for task in section["tasks"])
            or not isinstance(section.get("mini_project"), str)
        ):
            raise ValueError("Each roadmap period must include exactly three tasks.")
    if not isinstance(plan.get("todays_action"), str) or not plan["todays_action"].strip():
        raise ValueError("The plan is missing today's action.")
    return plan


# Check that Gemini returned ten unique, well-labeled questions.
def parse_interview_questions(answer):
    result = json.loads(answer)
    if not isinstance(result, dict) or not isinstance(result.get("questions"), list):
        raise ValueError("The questions must be returned as a list.")
    questions = result["questions"]
    if len(questions) != 10:
        raise ValueError("The response must include exactly ten questions.")

    seen_questions = set()
    has_simple = False
    has_intermediate = False
    for item in questions:
        if (
            not isinstance(item, dict)
            or set(item) != {"question", "difficulty"}
            or not isinstance(item["question"], str)
            or not item["question"].strip()
            or item["difficulty"] not in ("Simple", "Intermediate")
        ):
            raise ValueError("A question is missing its text or difficulty.")
        normalized_question = " ".join(item["question"].casefold().split()).rstrip("?.!")
        if normalized_question in seen_questions:
            raise ValueError("The response contains duplicate questions.")
        seen_questions.add(normalized_question)
        has_simple = has_simple or item["difficulty"] == "Simple"
        has_intermediate = has_intermediate or item["difficulty"] == "Intermediate"
    if not has_simple or not has_intermediate:
        raise ValueError("The response must include both difficulty levels.")
    return questions


def plan_as_markdown(plan):
    lines = ["# My Career Plan", "", "## Do this today", plan["todays_action"], "", "## Roles that fit me"]
    for role in plan["roles"]:
        lines.extend([
            "",
            f"### {role['title']}",
            f"**Match score:** {role['match_score']}%",
            f"**Why it fits:** {role['why_it_fits']}",
            f"**Indicative salary range:** {role['salary_range_lpa']} LPA",
        ])
    lines.extend(["", "## Skill gaps"])
    for gap in plan["skill_gaps"]:
        lines.append(
            f"- **{gap['skill']}**: {gap['current_level']}/10 now, "
            f"{gap['needed_level']}/10 target. Free resource: {gap['free_resource']}"
        )
    lines.extend(["", "## 90-day roadmap"])
    for title, key in [
        ("Days 1-30", "days_1_30"),
        ("Days 31-60", "days_31_60"),
        ("Days 61-90", "days_61_90"),
    ]:
        section = plan["roadmap"][key]
        lines.extend(["", f"### {title}"])
        lines.extend(f"- [ ] {task}" for task in section["tasks"])
        lines.extend(["", f"**Mini project:** {section['mini_project']}"])
    return "\n".join(lines)


st.set_page_config(page_title="AI Career Planner", page_icon="🎯", layout="centered")

# A little CSS (page styling). "centered" layout keeps everything in a middle
# column; we make that column a bit wider so the 3 role cards fit side by side.
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1000px; padding-top: 2.5rem; }
    .hero { text-align: center; margin-bottom: 1.5rem; }
    .hero h1 { font-size: 2.6rem; margin-bottom: 0.2rem; }
    .hero p { color: #475569; font-size: 1.1rem; margin: 0; }
    .step { color: #2563EB; font-weight: 600; font-size: 0.85rem;
            letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .score { font-size: 2.4rem; font-weight: 700; color: #2563EB; line-height: 1.1; }
    .score span { font-size: 1rem; font-weight: 500; color: #475569; }
    .footer { text-align: center; color: #64748B; font-size: 0.85rem; padding: 0.5rem 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Centered title and subtitle
st.markdown(
    """
    <div class="hero">
      <h1>🎯 AI Career Planner</h1>
      <p>My first AI product, built by HUDA PARVIN.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Step 1 lets people include a resume without saving it to disk.
st.markdown('<div class="step">Step 1 · Resume (optional)</div>', unsafe_allow_html=True)
resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
st.caption(
    "Your resume is sent to Google Gemini to make your plan and is not stored by "
    "this app. On Gemini's free tier Google may use it to improve its products."
)

# Step 2 collects the user's profile in the existing form.
with st.form("profile"):
    st.markdown('<div class="step">Step 2 · About you</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    name = left.text_input("Your name")
    education = right.selectbox(
        "Highest education",
        ["Plus Two / 12th", "Diploma", "B.Tech / B.E.", "B.Sc / BCA",
         "B.Com / BBA", "BA", "Master's degree", "Other"],
    )
    status = st.radio(
        "Where are you right now?",
        ["Student", "Graduate looking for a job", "Working, want to switch", "Career break"],
        horizontal=True,
    )
    left, right = st.columns(2)
    skills_label = "Skills you already have (optional with a resume)" if resume else "Skills you already have"
    skills = left.text_area(skills_label, placeholder="Excel, basic Python, communication...")
    interests = right.text_area("What do you enjoy?", placeholder="Solving puzzles, design, talking to people...")
    goal = st.text_input("Your goal", placeholder="Get my first job in tech within 6 months")
    hours = st.slider("Hours per week you can study", 2, 40, 10)
    submitted = st.form_submit_button("🚀 Plan my career", type="primary", width="stretch")

if submitted:
    missing = []
    if not name.strip():
        missing.append("**Your name**")
    if not goal.strip():
        missing.append("**Your goal**")
    if not skills.strip() and resume is None:
        missing.append("**Skills you already have**")
    if missing:
        st.warning("Please fill in: " + ", ".join(missing) + ".")
    else:
        # Send the profile and optional PDF to Gemini from memory.
        profile = {
            "name": name,
            "education": education,
            "status": status,
            "skills": skills,
            "interests": interests,
            "goal": goal,
            "hours_per_week": hours,
        }

        try:
            client = create_gemini_client()
            contents = [f"Here is my career profile:\n{json.dumps(profile)}"]
            if resume is not None:
                contents.append(
                    types.Part.from_bytes(data=resume.getvalue(), mime_type="application/pdf")
                )
            with st.spinner("Thinking about your career..."):
                answer = generate_text(
                    client,
                    contents,
                    SYSTEM_PROMPT,
                    response_schema=CAREER_PLAN_SCHEMA,
                )
            st.session_state["career_plan"] = parse_career_plan(answer)
            st.session_state["career_plan_version"] = st.session_state.get("career_plan_version", 0) + 1
            st.session_state["career_chat"] = []
        except KeyError:
            st.error("The Gemini API key is missing. Add it to .streamlit/secrets.toml and try again.")
        except errors.APIError as exc:
            if exc.code in (429, 500, 503, 504):
                st.error("Gemini is busy or has reached its free limit. Please wait a moment and try again.")
            elif exc.code in (401, 403):
                st.error("The Gemini API key is not accepted. Check that it is a valid key and try again.")
            elif exc.code == 404:
                st.error("The Gemini model is not available for this API key. Please try again later.")
            else:
                st.error("I couldn't create your career plan right now. Please try again in a moment.")
        except (json.JSONDecodeError, ValueError):
            st.error("Gemini returned a plan in an unexpected format. Please try again.")

# The dashboard reads the saved plan so it stays visible after reruns.
if "career_plan" in st.session_state and isinstance(st.session_state["career_plan"], dict):
    plan = st.session_state["career_plan"]
    st.success(f"**Do this today**\n\n{plan['todays_action']}")

    role_columns = st.columns(3)
    for column, role in zip(role_columns, plan["roles"]):
        with column:
            with st.container(border=True):
                st.subheader(role["title"])
                st.markdown(
                    f'<div class="score">{role["match_score"]}<span> / 100</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(role["match_score"])
                st.write(role["why_it_fits"])
                st.caption(f"Indicative salary: {role['salary_range_lpa']} LPA")

    st.subheader("Skill gaps")
    st.bar_chart(
        {
            "Skill": [gap["skill"] for gap in plan["skill_gaps"]],
            "Current level": [gap["current_level"] for gap in plan["skill_gaps"]],
            "Needed level": [gap["needed_level"] for gap in plan["skill_gaps"]],
        },
        x="Skill",
        y=["Current level", "Needed level"],
        horizontal=True,
    )
    for gap in plan["skill_gaps"]:
        st.markdown(f"- **{gap['skill']}** — Free resource: {gap['free_resource']}")

    st.subheader("Your 90-day roadmap")
    roadmap_tabs = st.tabs(["Days 1-30", "Days 31-60", "Days 61-90"])
    roadmap_periods = [
        "days_1_30",
        "days_31_60",
        "days_61_90",
    ]
    plan_version = st.session_state.get("career_plan_version", 0)
    for tab, period_key in zip(roadmap_tabs, roadmap_periods):
        with tab:
            section = plan["roadmap"][period_key]
            for index, task in enumerate(section["tasks"]):
                st.checkbox(task, key=f"task_{plan_version}_{period_key}_{index}")
            with st.container(border=True):
                st.markdown(f"**Mini project: {section['mini_project']}**")

    st.download_button(
        "Download my plan",
        data=plan_as_markdown(plan),
        file_name="my-career-plan.md",
        mime="text/markdown",
    )

    # Follow-up answers use the saved plan and stay under 150 words.
    st.subheader("Ask about your plan")
    with st.form("plan_chat"):
        question = st.text_input("Your question")
        ask_submitted = st.form_submit_button("Ask")

    if "career_chat" not in st.session_state:
        st.session_state["career_chat"] = []
    if ask_submitted and question.strip():
        try:
            client = create_gemini_client()
            chat_context = (
                "Answer the user's question using this career plan as context. "
                "Use simple English, do not promise a job or salary, and answer in under 150 words.\n\n"
                f"Career plan:\n{json.dumps(plan)}\n\nQuestion: {question.strip()}"
            )
            with st.spinner("Thinking about your question..."):
                answer = generate_text(
                    client,
                    chat_context,
                    "You are a helpful career counsellor.",
                )
            words = answer.split()
            if len(words) > 149:
                answer = " ".join(words[:149]) + "..."
            st.session_state["career_chat"].append((question.strip(), answer))
        except KeyError:
            st.error("The Gemini API key is missing. Add it to .streamlit/secrets.toml and try again.")
        except errors.APIError as exc:
            if exc.code in (429, 500, 503, 504):
                st.error("Gemini is busy or has reached its free limit. Please wait a moment and try again.")
            elif exc.code in (401, 403):
                st.error("The Gemini API key is not accepted. Check that it is a valid key and try again.")
            else:
                st.error("I couldn't answer that right now. Please try again in a moment.")
        except ValueError:
            st.error("Gemini returned an empty answer. Please try again.")

    for user_question, assistant_answer in st.session_state["career_chat"]:
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            st.write(assistant_answer)

# Generate and keep role-specific interview questions.
st.divider()
st.subheader("Interview Questions")
job_title = st.text_input(
    "Desired job title",
    placeholder="Python Developer, Data Analyst, Web Developer...",
    max_chars=120,
)
generate_questions = st.button("Generate Questions")
regenerate_questions = False
if st.session_state.get("interview_questions"):
    regenerate_questions = st.button("Regenerate Questions")

if generate_questions or regenerate_questions:
    if not job_title.strip():
        st.warning("Enter a job title before generating questions.")
    else:
        question_prompt = (
            "Create exactly 10 commonly asked interview questions for this job title: "
            f"{job_title.strip()}. Keep them relevant and beginner-friendly, from Simple "
            "to Intermediate difficulty. Include both technical and conceptual topics, "
            "covering important role-related skills, tools, technologies, and basic "
            "practical concepts. Avoid duplicates and advanced or expert-level topics. "
            "Return only the questions and their difficulty labels in the required JSON "
            "format. Do not include answers, explanations, hints, or solutions."
        )
        try:
            client = create_gemini_client()
            with st.spinner("Generating interview questions..."):
                answer = generate_text(
                    client,
                    question_prompt,
                    "You generate interview questions only. Follow the requested role, "
                    "difficulty limits, and JSON schema exactly.",
                    response_schema=INTERVIEW_QUESTIONS_SCHEMA,
                )
            st.session_state["interview_questions"] = parse_interview_questions(answer)
            st.session_state["interview_questions_job_title"] = job_title.strip()
        except KeyError:
            st.error("The Gemini API key is missing. Add it to .streamlit/secrets.toml and try again.")
        except errors.APIError as exc:
            if exc.code in (429, 500, 503, 504):
                st.error("Gemini is busy or has reached its free limit. Please wait a moment and try again.")
            elif exc.code in (401, 403):
                st.error("The Gemini API key is not accepted. Check that it is a valid key and try again.")
            elif exc.code == 404:
                st.error("The Gemini model is not available for this API key. Please try again later.")
            else:
                st.error("I couldn't generate interview questions right now. Please try again in a moment.")
        except (json.JSONDecodeError, ValueError):
            st.error("Gemini returned questions in an unexpected format. Please try again.")

if st.session_state.get("interview_questions"):
    st.caption(f"Questions for: {st.session_state['interview_questions_job_title']}")
    for number, item in enumerate(st.session_state["interview_questions"], start=1):
        st.markdown(f"{number}. **{item['difficulty']}**: {item['question']}")

# Footer
st.divider()
st.markdown(
    '<div class="footer">Built at the Techolas AI Wing weekend · AI suggestions, not guarantees.</div>',
    unsafe_allow_html=True,
)
