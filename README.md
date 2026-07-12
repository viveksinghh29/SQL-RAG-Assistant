#  SQL RAG Assistant

An enterprise-grade AI-powered SQL assistant that enables users to query relational databases using natural language. The system leverages Retrieval-Augmented Generation (RAG) to understand database schemas and business documentation, generates secure SQL queries, validates them, executes them against the database, and presents results with AI-generated explanations and interactive visualizations.

Built with **FastAPI**, **MySQL**, **LangChain**, **FAISS**, **Groq Llama 3**, and **Streamlit**.

---

## ✨ Features

* 💬 Natural Language to SQL
* 🔍 Retrieval-Augmented Generation (RAG)
* 🛡️ Secure SQL validation before execution
* 📊 AI-powered data visualization
* 📖 Natural language query explanations
* 🗂️ Schema-aware semantic search
* 🔐 JWT Authentication
* ⚡ Async FastAPI backend
* 🏗️ Clean Architecture & Repository Pattern
* 📈 Interactive Streamlit dashboard

---

##  Tech Stack

| Category           | Technologies                         |
| ------------------ | ------------------------------------ |
| **Backend**        | FastAPI, SQLAlchemy 2.0, Pydantic v2 |
| **Database**       | MySQL, Alembic                       |
| **AI**             | LangChain, Groq (Llama 3)            |
| **RAG**            | Sentence Transformers, FAISS         |
| **Frontend**       | Streamlit, Plotly                    |
| **Authentication** | JWT                                  |
| **Caching**        | Redis                                |


## Screenshot
1[image alt](https://github.com/viveksinghh29/SQL-RAG-Assistant/blob/main/image.png?raw=true)
1[image alt]()

## 📁 Project Structure

```text
sql-rag-assistant/
├── app/                # FastAPI backend
├── alembic/            # Database migrations
├── data/               # Sample data & vector indexes
├── docs/               # Project documentation
├── frontend/           # Streamlit application
├── scripts/            # Utility scripts
├── tests/              # Unit & integration tests
├── .env.example
├── alembic.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

##  Installation

```bash
git clone <repository-url>

cd sql-rag-assistant

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements-dev.txt

cp .env.example .env

# Configure API keys and database credentials

uvicorn app.main:app --reload
```

---

##  API

| Endpoint  | Description               |
| --------- | ------------------------- |
| `/docs`   | Swagger API Documentation |
| `/health` | Health Check              |

---

##  Workflow

```text
User Question
      │
      ▼
Retrieve Relevant Schema & Business Context (RAG)
      │
      ▼
Generate SQL with LLM
      │
      ▼
Validate SQL
      │
      ▼
Execute Query
      │
      ▼
Generate AI Explanation
      │
      ▼
Visualize Results
```

---

##  Future Enhancements

* Conversation Memory
* Multi-Database Support
* Query Optimization
* Cloud Deployment
* Docker Support
* CI/CD Pipeline
* Role-Based Access Control
* Streaming Responses

---

Author

Vivek Kumar Singh

Final Year Computer Science Student
AI • Machine Learning • Data Analytics • Full Stack Development


## 📄 License

MIT License
