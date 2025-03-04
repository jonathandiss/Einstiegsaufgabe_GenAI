from transformers import BartTokenizer, BartForConditionalGeneration


model_path = "trained_model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

text = (
    "President Donald Trump's decision to pause all US aid to Ukraine is a bitter blow – not just for Kyiv but also European allies who have been lobbying the US administration to continue its support.\n"
    "\n"
    "This is not the first time that the US has withheld military aid. Republicans in Congress blocked then-President Joe Biden's largest tranche of military assistance for Ukraine in the summer of 2023.\n"
    "\n"
    "Then, Ukraine just about managed to eke out its existing stocks of ammunition with the help of Europe.\n"
    "\n"
    "Congress finally approved the £60bn aid package in the spring of 2024. It was just in time - Ukraine was struggling to fend off a renewed Russian offensive in Kharkiv. The arrival of the delayed US weapons helped turn the tide.\n"
    "\n"
    "As in 2024, it may be months before the effects of cutting off US aid are felt – at least in terms of ammunition and hardware. European nations have slowly ramped up their production of artillery shells. Overall Europe now provides Ukraine with 60 percent of its aid – more than the US.\n"
    "\n"
    "However, US military support is still vital to Ukraine. One Western official recently described it as \"the cream\" in terms of weapons.\n"
    "\n"
    "Ukraine's ability to protect its people and cities has heavily relied on sophisticated US air defence systems – such as Patriot batteries and NASAMS – jointly developed with Norway.\n"
    "\n"
    "The US has given Ukraine the ability to carry out long range strikes – with HIMARS and ATACM missiles. The US has limited their use inside Russia, but they have still been vital to hit high value targets inside occupied territories.\n"
    "\n"
    "It is not just quality, but quantity too. As the world's most powerful military, it has been able to send hundreds of surplus Humvees and armoured vehicles - numbers that smaller European armies could never match.\n"
    "\n"
    "The absence of some of this aid may take time to filter down to the frontline. But there could be a more worrying immediate impact, not least in terms of intelligence sharing.\n"
    "\n"
    "No nation can match the US in terms of space-based surveillance, intelligence gathering and communications. And it is not just provided by the US military, but commercial companies too."
)

print({text})

inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
print(inputs)

summary = model.generate(inputs["input_ids"], max_length=64, num_beams=4,early_stopping=True)
print(summary)

title = tokenizer.decode(summary[0], skip_special_tokens=True)
print(title)