# CHECKPOINT 1 — "Hello, AI builder"
# This is the file every attendee starts with. If you can see this page
# in the preview on the right, your setup is done.
# It uses the same page layout as the final app, so the look never changes.

import streamlit as st
from google import genai
from google.genai import errors, types

# Prompt for Gemini
SYSTEM_PROMPT = """
You are a friendly and honest career counsellor for students and job seekers in India.
Given a person's profile, reply in Markdown with exactly these sections:
## 3 roles that fit you
For each role, give the role name, why it fits the person, and an indicative salary range in LPA.
## Your skill gaps
Give a table with columns: Skill, Current level, Level to aim for, and Free resource.
## Your 90-day roadmap
Split the plan into Days 1-30, Days 31-60, and Days 61-90. Give 3 practical tasks for each period.
## One thing to do today
Suggest one small task that can be completed in under 30 minutes.
Rules: Never promise a job or a fixed salary. Label every salary as indicative. Use simple English.
Keep the answer under 450 words and give advice that is realistic for a beginner.
"""

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

# The profile form, laid out in two columns
with st.form("profile"):
    st.markdown('<div class="step">About you</div>', unsafe_allow_html=True)
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
    skills = left.text_area("Skills you already have", placeholder="Excel, basic Python, communication...")
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
    if not skills.strip():
        missing.append("**Skills you already have**")
    if missing:
        st.warning("Please fill in: " + ", ".join(missing) + ".")
    else:
        # Send the completed profile to Gemini
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
            client = genai.Client(
                api_key=st.secrets["GEMINI_API_KEY"],
                http_options=types.HttpOptions(api_version="v1beta"),
            )
            answer = None
            last_error = None
            models = [
                ("gemini-3.5-flash-lite", 2),
                ("gemini-2.5-flash-lite", 1),            ]

            for model, attempts in models:
                for attempt in range(attempts):
                    try:
                        with st.spinner("Thinking about your career..."):
                            response = client.models.generate_content(
                                model=model,
                                contents=f"Here is my career profile:\n{profile}",
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    temperature=0.7,
                                ),
                            )
                        answer = response.text
                        break
                    except errors.APIError as exc:
                        last_error = exc
                        if exc.code not in (404, 429, 500, 503, 504):
                            raise
                if answer:
                    break

            if not answer:
                if isinstance(last_error, errors.APIError):
                    raise last_error
                raise RuntimeError(str(last_error or "Gemini returned no answer"))

            st.session_state["career_plan"] = answer
            with st.container(border=True):
                st.markdown(answer)
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
        except Exception:
            st.error("I couldn't create your career plan right now. Please try again in a moment.")

# Keep a successful plan visible after Streamlit reruns
if not submitted and "career_plan" in st.session_state:
    with st.container(border=True):
        st.markdown(st.session_state["career_plan"])

# Footer
st.divider()
st.markdown(
    '<div class="footer">Built at the Techolas AI Wing weekend · AI suggestions, not guarantees.</div>',
    unsafe_allow_html=True,
)
