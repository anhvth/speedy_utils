from llm_utils.lm import LLM

lm = LLM(client=8000)

# Encode text to tokens
token_ids = lm.encode('Hello, world!')
print(f'Token IDs: {token_ids}')

# Decode tokens back to text
text = lm.decode(token_ids)
print(f'Decoded text: {text}')

# Get token strings for debugging
ids, strs = lm.encode('Hello', return_token_strs=True)
print(f'IDs: {ids}')
print(f'Strings: {strs}')
print(f'Tokens with strings:')
for i, s in zip(ids, strs):
    print(f'  {i:6d} -> "{s}"')
