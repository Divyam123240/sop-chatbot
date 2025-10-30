# SOP Chatbot (Ollama + PostgreSQL + pgvector)

Default Ollama model set to: llama3

## Quickstart
1. Start Postgres:
   docker-compose up -d db
2. Backend:
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
   export OLLAMA_URL=http://localhost:11434
   export OLLAMA_MODEL=llama3
   uvicorn main:app --reload --port 8000
3. Frontend:
   cd frontend
   npm install
   npm run dev
Open http://localhost:5173
