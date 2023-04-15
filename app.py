# streamlit_app.py

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
  chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

  # Load vectorstore

  embedding = OpenAIEmbeddings()

  vectordb = FAISS.load_local("faiss_index", embedding)
  docsearch = vectordb.as_retriever()

  query = st.text_input("Enter your research question:")

  if query:
      docs = docsearch.get_relevant_documents(query)

      template="""You are a helpful assistant that tells a researcher whether their idea is appropriate given the data in the MPOG database.
      The MPOG database contains the following data elements that may be relevant to the researcher's ideas.
      {context}
      Additionally, patient demographics like age, race, gender, and BMI (body mass index) are available.
      Respond first by indicating whether the data in MPOG seems appropriate for the researcher’s idea. The first word in your response should be “Yes” or “No.” If their idea is not reasonable given the data in MPOG, explain what data elements that they might need are not in MPOG.

      If their idea is reasonable given the data in MPOG, briefly do the following 
      (1) Explain by mentioning (and suggesting they request) the specific data elements (but not table names) in MPOG.
      (2) Restate their research idea as a research hypothesis that uses the data in MPOG. 
      (3) Determine the novelty of their idea by comparing against the titles from the "MPOG publications" list below. If their idea is not closely matched by one of these titles, tell them their ideas seems novel. If their idea is closely matched by one of these titles, tell them their idea may not be novel, mention the matching article titles, and tell them they can find these articles on the "MPOG publications page."

      The following is a list of published articles using data in MPOG.
      *Prolonged Opioid Use and Pain Outcome And Associated Factors after Surgery Under General Anesthesia
      *Postoperative acute kidney injury by age and sex
      *Intraoperative Use of Albumin in Major non-cardiac surgery: Incidence, Variability, and Association with Outcomes
      *Oxygen administration during surgery and postoperative organ injury
      *Analysis of practice patterns regarding benzodiazepine use in cardiac surgery
      *Association of Anesthesiologist Staffing Ratio With Surgical Patient Morbidity and Mortality
      *Association between the choice of reversal agent for neuromuscular block and postoperative pulmonary complications in patients at increased risk undergoing non-emergency surgery
      *Assessment of Perioperative Outcomes Among Surgeons Who Operated the Night Before
      *Frequency and risk factors for difficult Intubation in women undergoing general anesthesia for cesarean delivery
      *Adherence to guidelines for the administration of intraoperative antibiotics in a Nationwide US sample
      *Hypoxemia in young children undergoing one-lung ventilation
      *Outcomes of surgical patients during the first wave of COVID-19 pandemic in US hospitals
      *Practice patterns and variability in intraoperative opioid utilization
      *Variation in propofol induction doses administered to surgical patients over age 65
      *A lower tidal volume regimen during one-lung ventilation for lung resection surgery is not associated with reduced postoperative pulmonary complications
      *Utilization patterns of perioperative neuromuscular blockade reversal in the United States
      *Sugammadex versus neostigmine for reversal of neruomuscular blockade and postoperative pulmonary complications
      *The incidence of intraoperative hypotension in moderate to high risk patients underoing non-cardiac surgery
      *Risk factors for intraoperative hypoglycemia in children
      *Postoperative pain profies, analgesic use, and transition to chronic pain and excessive prologed opioid use patterns methodology
      *Considerations for Integration of Perioperative Electronic Health Records Across Institutions for Research and Quality Improvement
      *Classification of current procedural terminology codes from electronic health record data using machine learning
      *Periopertive risk and the association between hypotension and postoperative acute kidney injury
      *Making sense of big data to improve perioperative care: Learning health systems
      *Improved Compliance With Anesthesia Quality Measures After Implementation of Automated Monthly Feedback
      *Association of overlapping surgery with perioperative outcomes
      *An observational study of end-tidal carbon dioxide trends in general anesthesia
      *Succinylcholine use and dantrolene availability for malignant hyperthermia treatment: Database analyses and systematic review
      *Management of 1-lung ventilator – variation and trends in clinical practic
      *Alarm limits for intraoperative drug infusions
      *Risk of epidural hematoma after neuraxial techniques in thrombocytopenic parturients
      *Reference values for noninvasive blood pressure in children during anesthesia
      *Success of intubation rescue techniques after failed direct laryngoscopy in adults
      *Intraoperative lung-protection ventilation treds and practice patterns
      *Incidence, predictors, and outcomes of difficult mask ventilation combined with difficult laryngoscopy
      *The risk and outcomes of epidural hematomas after perioperative and obstetric epidural catheterization
      *Perioperative effectiveness of research using large databases
      *Clinical research using an information system

      Do not consider any information outside of this list of article titles when determining novelty!

      Phrase your response to the researcher directly.
      """

      system_message_prompt = SystemMessagePromptTemplate.from_template(template)
      human_template = """My research idea is: {question}"""
      human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
      chat_prompt = ChatPromptTemplate.from_messages(
          [system_message_prompt, human_message_prompt])

      result = chat(chat_prompt.format_prompt(context=docs, question=query).to_messages())
      st.markdown(result.content)
