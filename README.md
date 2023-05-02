# mpog-helper-openai
Uses OpenAI API to determine if a research idea is reasonable and novel for the MPOG database. Not affiliated with MPOG in any way.

## Deploy
Assuming your OpenAI API key is in .streamlit/secrets.toml
```
docker build -t streamlit .
docker create --name mpog-web -p 8501:8501 streamlit
docker cp ../.streamlit/secrets.toml mpog-web:/deploy/.streamlit/
docker start mpog-web
```