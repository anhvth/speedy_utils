"""
Beautiful example script for interacting with VLLM server.

This script demonstrates various ways to use the VLLM API server
for text generation tasks.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


class VLLMRequest(BaseModel):
    """Request model for VLLM API."""
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class VLLMResponse(BaseModel):
    """Response model from VLLM API."""
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class VLLMClient:
    """Client for interacting with VLLM server."""
    
    def __init__(self, base_url: str = 'http://localhost:8140'):
        self.base_url = base_url
        self.model_name = 'selfeval_8b'
        
    async def generate_text(
        self, 
        request: VLLMRequest
    ) -> VLLMResponse:
        """Generate text using VLLM API."""
        url = f'{self.base_url}/v1/completions'
        
        payload = {
            'model': self.model_name,
            'prompt': request.prompt,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'stream': request.stream,
        }
        
        if request.stop:
            payload['stop'] = request.stop
            
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    choice = data['choices'][0]
                    usage = data['usage']
                    
                    return VLLMResponse(
                        text=choice['text'],
                        finish_reason=choice['finish_reason'],
                        prompt_tokens=usage['prompt_tokens'],
                        completion_tokens=usage['completion_tokens'],
                        total_tokens=usage['total_tokens']
                    )
                    
            except aiohttp.ClientError as e:
                logger.error(f'HTTP error: {e}')
                raise
            except Exception as e:
                logger.error(f'Unexpected error: {e}')
                raise
                
    async def generate_batch(
        self, 
        requests: List[VLLMRequest]
    ) -> List[VLLMResponse]:
        """Generate text for multiple requests concurrently."""
        tasks = [self.generate_text(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    async def health_check(self) -> bool:
        """Check if the VLLM server is healthy."""
        url = f'{self.base_url}/health'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f'Health check failed: {e}')
            return False


async def example_basic_generation():
    """Example: Basic text generation."""
    logger.info('🚀 Running basic generation example')
    
    client = VLLMClient()
    
    # Check server health
    if not await client.health_check():
        logger.error('❌ Server is not healthy')
        return
        
    request = VLLMRequest(
        prompt='Explain the concept of machine learning in simple terms:',
        max_tokens=256,
        temperature=0.7,
        stop=['\n\n']
    )
    
    try:
        response = await client.generate_text(request)
        
        logger.success('✅ Generation completed')
        logger.info(f'📝 Generated text:\n{response.text}')
        logger.info(f'📊 Tokens: {response.total_tokens} total '
                   f'({response.prompt_tokens} prompt + '
                   f'{response.completion_tokens} completion)')
                   
    except Exception as e:
        logger.error(f'❌ Generation failed: {e}')


async def example_batch_generation():
    """Example: Batch text generation."""
    logger.info('🚀 Running batch generation example')
    
    client = VLLMClient()
    
    prompts = [
        'What is artificial intelligence?',
        'Explain quantum computing briefly:',
        'What are the benefits of renewable energy?'
    ]
    
    requests = [
        VLLMRequest(
            prompt=prompt,
            max_tokens=128,
            temperature=0.8
        ) for prompt in prompts
    ]
    
    try:
        responses = await client.generate_batch(requests)
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f'❌ Request {i+1} failed: {response}')
            else:
                logger.success(f'✅ Request {i+1} completed')
                logger.info(f'📝 Response {i+1}:\n{response.text}\n')
                
    except Exception as e:
        logger.error(f'❌ Batch generation failed: {e}')


async def example_creative_writing():
    """Example: Creative writing with specific parameters."""
    logger.info('🚀 Running creative writing example')
    
    client = VLLMClient()
    
    request = VLLMRequest(
        prompt=(
            'Write a short story about a robot discovering emotions. '
            'The story should be exactly 3 paragraphs:\n\n'
        ),
        max_tokens=400,
        temperature=1.2,  # Higher temperature for creativity
        top_p=0.95,
        stop=['THE END', '\n\n\n']
    )
    
    try:
        response = await client.generate_text(request)
        
        logger.success('✅ Creative writing completed')
        logger.info(f'📚 Story:\n{response.text}')
        logger.info(f'🎯 Finish reason: {response.finish_reason}')
        
    except Exception as e:
        logger.error(f'❌ Creative writing failed: {e}')


async def example_code_generation():
    """Example: Code generation."""
    logger.info('🚀 Running code generation example')
    
    client = VLLMClient()
    
    request = VLLMRequest(
        prompt=(
            'Write a Python function that calculates the fibonacci '
            'sequence up to n terms:\n\n```python\n'
        ),
        max_tokens=300,
        temperature=0.2,  # Lower temperature for code
        stop=['```', '\n\n\n']
    )
    
    try:
        response = await client.generate_text(request)
        
        logger.success('✅ Code generation completed')
        logger.info(f'💻 Generated code:\n```python\n{response.text}\n```')
        
    except Exception as e:
        logger.error(f'❌ Code generation failed: {e}')


async def main():
    """Run all examples."""
    logger.info('🎯 Starting VLLM Client Examples')
    logger.info('=' * 50)
    
    examples = [
        example_basic_generation,
        example_batch_generation,
        example_creative_writing,
        example_code_generation
    ]
    
    for example in examples:
        await example()
        logger.info('-' * 50)
        await asyncio.sleep(1)  # Brief pause between examples
        
    logger.info('🎉 All examples completed!')


if __name__ == '__main__':
    # Configure logger
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=''),
        format='<green>{time:HH:mm:ss}</green> | '
               '<level>{level: <8}</level> | '
               '<cyan>{message}</cyan>',
        level='INFO'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('\n👋 Goodbye!')
    except Exception as e:
        logger.error(f'❌ Script failed: {e}')
