from keras.models import load_model
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from train import CHUNK_SIZE
import numpy as np
import os


def generate_output(model, start_index=None, diversity=1.0, amount=400):
    if start_index is None:
        start_index = random.randint(0, len(training_text) - CHUNK_SIZE - 1)
    generated = training_text[start_index: start_index + CHUNK_SIZE]
    yield generated + "#" # Hash shows where initial fragment ends
    for i in range(amount):
        x = np.zeros((1, CHUNK_SIZE, len(chars)))
        for t, char in enumerate(generated):
            x[0, t, char_to_idx[char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        if diversity is None:
            next_index = np.argmax(preds[len(generated) - 1])
        else:
            preds = np.asarray(preds[len(generated) - 1]).astype('float64')
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            next_index = np.argmax(probas)
        next_char = chars[next_index]
        yield next_char
        generated += next_char
    return generated

if __name__ == "__main__":
    shakespeare = strip_headers(load_etext(100))
    training_text = shakespeare.split("\nTHE END", 1)[-1]
    print("Number of entries: ", len(training_text))

    BASE = os.path.dirname(os.path.abspath(__file__))
    MODEL = os.path.join(BASE, "models", "")

    for ch in generate_output(model, training_text):
        sys.stdout.write(ch)
    print()

