# 🧠 LLM-API Flask Server

A **production-ready API server** built using **Flask**, **Hugging Face Transformers**, and **PyTorch** — designed to deploy a **local Large Language Model (LLM)** with support for both **GPU (CUDA)** and **CPU** environments.  

The project comes fully **Dockerized** for easy setup, testing, and deployment in any environment.

---

## 🚀 Features

- ⚡ REST API endpoints for LLM-based text generation  
- 🧩 GPU (CUDA) acceleration with automatic fallback to CPU  
- 🐳 Docker & Docker Compose support for quick deployment  
- 🧱 Modular structure — easy to extend, customize, and scale  
- 💾 Persistent model caching for faster subsequent runs  

---

## 📂 Project Structure

```

.
├── api.py              # Flask API for HTTP endpoints
├── llm.py              # Model loading and inference logic
├── requirements.txt    # Python and library dependencies
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Multi-container setup with GPU support
└── README.md           # Project documentation

````

---

## ⚙️ Installation

### 🧩 Prerequisites

- [Docker](https://www.docker.com/)  
- (Optional) **NVIDIA GPU drivers** and **Docker’s NVIDIA runtime** for GPU acceleration  

---

### 🐍 Setup

Clone the repository:

```bash
git clone https://github.com/rajvansh-369/llm_chatbot_api.git
cd llm_chatbot_api
````

(Optional) Edit `llm.py` or `api.py` to change model settings or API routes.

---

## 🧱 Running the API (Docker Compose)

Build and start the containers:

```bash
docker compose up --build
```

Once running, your API will be accessible at:

👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🌐 API Endpoints

### 1. **Greet Endpoint**

**Request:**

```bash
GET /greet?name=John
```

**Response:**

```html
<h1>Hello, John!</h1>
```

---

### 2. **LLM Input Endpoint**

**Request:**

```bash
POST /input
Content-Type: application/json

{
  "prompt": "Who won the FIFA World Cup in 2022?"
}
```

**Response:**

```json
{
  "prompt": "Who won the FIFA World Cup in 2022?",
  "response": "<model's generated answer>",
  "status": "success"
}
```

---

## 🧠 Development

* Modify `llm.py` to switch models or adjust generation parameters.
* Edit `api.py` to add or modify REST API endpoints.
* The first run may take longer as the model downloads and caches locally.

🗂 Cached model weights are stored in the `./cache` directory (mapped in `docker-compose.yml`).

---

## ⚡ GPU Support

For GPU acceleration:

* Ensure **CUDA** is installed and configured on your system.
* Use Docker with **NVIDIA runtime**.
* Confirm GPU visibility inside the container using:

```bash
docker compose exec <service_name> nvidia-smi
```

---

## 🧾 Example Output

```bash
curl -X POST http://localhost:5000/input \
-H "Content-Type: application/json" \
-d '{"prompt": "Explain quantum computing in simple terms."}'
```

**Response:**

```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "response": "Quantum computing uses qubits, which can represent 0 and 1 simultaneously...",
  "status": "success"
}
```

---

## 🧩 Tech Stack

* [Flask](https://flask.palletsprojects.com/) – Lightweight Python web framework
* [Transformers](https://huggingface.co/transformers/) – State-of-the-art NLP models
* [PyTorch](https://pytorch.org/) – Deep learning framework
* [Docker](https://www.docker.com/) – Containerized deployment

---

## 🏗️ Future Improvements

* ✅ Add authentication (API key or JWT)
* ✅ Add streaming responses for long text generation
* ✅ Add async inference for faster parallel requests
* ✅ Add WebSocket endpoint for real-time interaction

---

## 🙌 Acknowledgments

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch](https://pytorch.org/)
* [Flask](https://flask.palletsprojects.com/)
* [Docker](https://www.docker.com/)

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it for your own projects.

---

### 👤 Author

**Snehal Rajvansh**
🔗 [GitHub](https://github.com/rajvansh-369)
📧 [Contact](mailto:rajbansh.snehal@gmail.com)


