# DataFlow Architect – Run Instructions

1. **Load the Docker image:**

   ```bash
   docker load < dataflow-architect.tar```

2. **Make sure you have a .env file with the OpenAI key**

3. **Run the application:**
   ```bash
   docker run -p 8501:8501 --env-file .env dataflow-architect```

4. **Access the application at http://localhost:8501/**
