from transformers import BartTokenizer, BartForConditionalGeneration


model_path = "trained_model"
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained(model_path)

text = (
    "It got cold overnight, I look for my gloves. Searching for gloves like when I was a child, I remember.Don't forget your gloves - I rummage nervously in my school bag and only find one glove. I don't want to look any further, there's no snow today on Christmas Eve. Disappointed about this, I later walk beside my grandma on the well-trodden country lane over the bridge towards the town and past the dark little grove, which I think could look as bright and decorated with snow as with lights. I pray silently - please let it snow.I sing all the songs in the church loudly and from the bottom of my heart, please let it snow.I don't take my eyes off the round church window above the altar - it's going to snow.As the bells ring in the Christmas night, the miracle - dancing snowflakes like feathers from angel wings in front of the round window.I jump up from the pew laughing and cheering - snow, snow!I had to explain my appearance in the reverent surroundings later.A little sad today, I put on my gloves, which I have found after all. I have lost my faith in miracles - finding them again could work."
)

input_ids = tokenizer.encode(
    text,
    return_tensors="pt",
    max_length=1024,
    truncation=True,
)

summary_text_ids = model.generate(
    input_ids=input_ids,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    max_length=142,
    min_length=56,
    num_beams=4,
)

decoded_text = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
print(decoded_text)