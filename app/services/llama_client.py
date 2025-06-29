"""Клиент для взаимодействия с LLaMA сервером."""

import asyncio
import json
from typing import Dict, Any, Optional
import httpx
from functools import lru_cache

from app.core.config import settings


class LlamaClient:
    """Клиент для взаимодействия с LLaMA сервером."""
    
    def __init__(self):
        self.base_url = settings.llama.server_url
        self.timeout = settings.llama.timeout
        self.max_tokens = settings.llama.max_tokens
        self.temperature = settings.llama.temperature
        
        # Настройки HTTP клиента
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход."""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния LLaMA сервера."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {
                "status": "healthy",
                "response_time": response.elapsed.total_seconds(),
                "server_info": response.json() if response.content else {}
            }
        except httpx.RequestError as e:
            return {
                "status": "unhealthy",
                "error": f"Connection error: {str(e)}"
            }
        except httpx.HTTPStatusError as e:
            return {
                "status": "unhealthy",
                "error": f"HTTP error: {e.response.status_code}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Unexpected error: {str(e)}"
            }
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list] = None
    ) -> Dict[str, Any]:
        """Генерация ответа от LLaMA модели."""
        
        # Подготавливаем параметры запроса
        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stop": stop_sequences or ["</s>", "<|end|>", "<|endoftext|>"]
        }
        
        # Добавляем system prompt если указан
        if system_prompt:
            # Формируем полный промпт с системными инструкциями
            full_prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{prompt}\n\n<|assistant|>\n"
            request_data["prompt"] = full_prompt
        
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "content": result.get("choices", [{}])[0].get("text", "").strip(),
                "usage": result.get("usage", {}),
                "model": result.get("model", "unknown"),
                "response_time": response.elapsed.total_seconds()
            }
            
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "content": ""
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP error: {e.response.status_code} - {e.response.text}",
                "content": ""
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON decode error: {str(e)}",
                "content": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "content": ""
            }
    
    async def generate_chat_completion(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Генерация ответа в формате чата."""
        
        request_data = {
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stream": False
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "content": result.get("choices", [{}])[0].get("message", {}).get("content", "").strip(),
                "usage": result.get("usage", {}),
                "model": result.get("model", "unknown"),
                "response_time": response.elapsed.total_seconds()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }
    
    async def embed_text(self, text: str) -> Dict[str, Any]:
        """Получение эмбеддинга текста через LLaMA сервер (если поддерживается)."""
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": text},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "embedding": result.get("data", [{}])[0].get("embedding", []),
                "usage": result.get("usage", {}),
                "response_time": response.elapsed.total_seconds()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "embedding": []
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "models": result.get("data", []),
                "response_time": response.elapsed.total_seconds()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "models": []
            }


# Глобальный экземпляр клиента
@lru_cache()
def get_llama_client() -> LlamaClient:
    """Получение глобального экземпляра LLaMA клиента."""
    return LlamaClient() 