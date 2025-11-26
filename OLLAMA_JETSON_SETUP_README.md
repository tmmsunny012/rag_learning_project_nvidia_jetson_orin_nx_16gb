# Running Ollama on NVIDIA Jetson Orin (Docker + Network API)

This guide explains how to run **Ollama** on the **NVIDIA Jetson Orin** (Seeed Studio reComputer J401) using **Docker**, expose the LLM REST API over the network, and test it from a Windows machine.

---

## Features
- GPU-accelerated Ollama using NVIDIA Container Runtime  
- Runs as a background Docker service  
- Accessible from LAN on port **11434**  
- Persistent model storage  
- Windows, Linux, macOS compatible clients  
- Optional Open WebUI  

---

## 1. Requirements
- NVIDIA Jetson Orin (Orin NX 16GB recommended)  
- Ubuntu 22.04 (JetPack 6.2 / L4T 36.4.x)  
- Docker installed  
- NVIDIA container runtime installed  

---

## 2. Install NVIDIA Container Runtime
```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 3. Test GPU Access
```bash
docker run --rm --runtime nvidia nvidia/cuda:12.0-base nvidia-smi
```

---

## 4. Create Model Storage Directory
```bash
mkdir -p ~/ollama
```

---

## 5. Run Ollama as Background Docker Service
```bash
docker run --runtime nvidia --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --env OLLAMA_HOST=0.0.0.0 --network host --shm-size=8g --volume /home/tmmsunny/ollama:/root/.ollama --name ollama -d dustynv/ollama:0.6.8-r36.4-cu126-22.04 ollama serve
```

---

## 6. Verify
```bash
docker ps
ss -tulpn | grep 11434
```

Expected:
```
LISTEN 0 4096 0.0.0.0:11434
```

---

## 7. Pull a Model
```bash
docker exec -it ollama ollama pull llama3.2:3b
```

---

## 8. Test API from Windows PowerShell
```powershell
(Invoke-WebRequest -Uri "http://192.168.178.124:11434/api/generate" -Method POST -Body '{"model":"llama3.2:3b","prompt":"hello from windows"}' -ContentType "application/json").Content
```

---

## 9. Optional: Open WebUI
```bash
docker run -d --network host --name openwebui -e OLLAMA_API_BASE_URL=http://localhost:11434 ghcr.io/open-webui/open-webui:main
```

Then open:
```
http://192.168.178.124:8080
```

---

## 10. Managing Containers
```
docker stop ollama
docker start ollama
docker logs ollama
docker rm ollama
```

---

## 11. Troubleshooting

### "model not found"
Pull the model:
```bash
docker exec -it ollama ollama pull llama3.2:3b
```

### Port not open
Check:
```bash
ss -tulpn | grep 11434
```

### Container stops immediately
Ensure:
```bash
ollama serve
```
is used as the container command.

---

## Completed

Ollama is now running as a full AI inference server on your Jetson Orin. 
