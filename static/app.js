const form = document.querySelector('#generator-form');
const result = document.querySelector('#result');
const status = document.querySelector('#status');
const submit = document.querySelector('#generate-button');
const copy = document.querySelector('#copy-button');
let outputText = '';

for (const id of ['tokens', 'temperature', 'top-k']) {
  const input = document.getElementById(id);
  const label = document.getElementById(`${id}-value`);
  input.addEventListener('input', () => { label.value = input.value; });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submit.disabled = true;
  copy.disabled = true;
  status.textContent = 'Generating…';
  result.replaceChildren();
  const waiting = document.createElement('p');
  waiting.className = 'empty';
  waiting.textContent = 'The model is choosing tokens…';
  result.append(waiting);
  const prompt = document.querySelector('#prompt').value;
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt,
        tokens: Number(document.querySelector('#tokens').value),
        temperature: Number(document.querySelector('#temperature').value),
        top_k: Number(document.querySelector('#top-k').value),
        seed: document.querySelector('#seed').value
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Generation failed.');
    outputText = data.text;
    const original = document.createElement('span');
    original.className = 'prompt-text';
    original.textContent = prompt;
    const continuation = document.createElement('span');
    continuation.className = 'generated-text';
    continuation.textContent = data.continuation;
    result.replaceChildren(original, continuation);
    status.textContent = `${data.generated_tokens} tokens generated`;
    copy.disabled = false;
  } catch (error) {
    result.textContent = error.message;
    status.textContent = 'Could not generate';
  } finally {
    submit.disabled = false;
  }
});

copy.addEventListener('click', async () => {
  await navigator.clipboard.writeText(outputText);
  copy.textContent = 'Copied';
  setTimeout(() => { copy.textContent = 'Copy text'; }, 1500);
});
