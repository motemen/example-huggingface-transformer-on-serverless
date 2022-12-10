import json
import os

from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-small")
tokenizer.do_lower_case = True # due to some bug of tokenizer config loading

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small")

def lambda_handler(event, context):
    body = json.loads(event['body'])

    input_text = body['text']
    inputs = tokenizer.encode_plus(input_text, return_tensors='pt')

    out = model.generate(**inputs, do_sample=True, num_return_sequences=3, max_new_tokens=30)

    results = []
    for generated in tokenizer.batch_decode(out, skip_special_tokens=True):
        results.append(generated)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'results': results
        })
    }
