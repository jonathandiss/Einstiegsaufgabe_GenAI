from transformers import BartTokenizer, BartForConditionalGeneration


model_path = "trained_model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

text = (
    "President Donald Trump's decision to pause all US aid to Ukraine is a bitter blow – not just for Kyiv but also European allies who have been lobbying the US administration to continue its support. This is not the first time that the US has withheld military aid. Republicans in Congress blocked then-President Joe Biden's largest tranche of military assistance for Ukraine in the summer of 2023. Then, Ukraine just about managed to eke out its existing stocks of ammunition with the help of Europe."
)

print({text})

# inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
# print(inputs)
#
# summary = model.generate(inputs["input_ids"], max_length=64, num_beams=5, early_stopping=True)
# print(summary)
#
# title = tokenizer.decode(summary[0], skip_special_tokens=True)
# print(title)

input_ids = tokenizer.encode(
    text,
    return_tensors="pt",
    max_length=1024,
    truncation=True,
)
print(input_ids)

summary_text_ids = model.generate(
    input_ids=input_ids,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    max_length=512,
    num_beams=4,
)
print(summary_text_ids)

decoded_text = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
print(decoded_text)