"""Interactive text generation helpers for local and pretrained chat backends."""

import re

import torch
from rich import print as rich_print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    chat_backend,
    code_max_new_tokens,
    code_model_name,
    conversation_turns,
    device,
    max_new_tokens,
    pretrained_max_new_tokens,
    pretrained_model_name,
    repetition_penalty,
    temperature,
    top_k,
    top_p,
)
from dataset import decode, encode, stoi

console = Console()
SYSTEM_PROMPT = (
    "You are a calm, clear chatbot. Respond naturally, stay on topic, "
    "and keep answers concise unless the user asks for detail."
)
CODE_SYSTEM_PROMPT = (
    "You are a careful coding assistant. Write correct Python and other code when asked. "
    "Prefer complete runnable examples, use fenced code blocks, explain briefly, and do not "
    "invent unavailable APIs."
)
CODE_REQUEST_PATTERN = re.compile(
    r"\b(code|python|javascript|typescript|java|c\+\+|c#|rust|go|sql|html|css|bash|shell|script|function|class|bug|debug|program|algorithm|regex|query|api|json|yaml|docker|flask|django|fastapi|react|node)\b",
    re.IGNORECASE,
)


def _normalize_reply_text(reply):
    normalized_reply = reply.replace("\r\n", "\n").strip()
    normalized_reply = re.sub(r"\n{3,}", "\n\n", normalized_reply)
    return normalized_reply or "I need a bit more context to answer that clearly."


def _render_reply(reply):
    console.print(
        Panel(
            Markdown(reply, code_theme="monokai", justify="left"),
            title="Bot",
            border_style="green",
            expand=True,
        )
    )


class LocalChatbot:
    def __init__(self, model):
        self.model = model
        self.history = []
        self.model.eval()

    def reply(self, user_message):
        reply = generate_reply(self.model, self.history, user_message)
        self.history.append((user_message, reply))
        return reply


class PretrainedChatbot:
    def __init__(self, model_name, system_prompt=SYSTEM_PROMPT, max_response_tokens=None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_response_tokens = max_response_tokens or pretrained_max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def reply(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        trimmed_messages = [self.messages[0], *self.messages[-(conversation_turns * 2):]]
        model_inputs = self.tokenizer.apply_chat_template(
            trimmed_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_response_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response_tokens = generated_ids[:, input_ids.shape[-1]:]
        reply = self.tokenizer.decode(response_tokens[0], skip_special_tokens=True).strip()
        if not reply:
            self.messages.pop()
            return "I need a bit more context to answer that clearly."
        cleaned_reply = _normalize_reply_text(reply)
        self.messages.append({"role": "assistant", "content": cleaned_reply})
        return cleaned_reply


class RoutedChatbot:
    def __init__(self, general_model_name, code_model_name):
        self.general_bot = PretrainedChatbot(general_model_name)
        self.code_model_name = code_model_name
        self.code_bot = None

    def _is_code_request(self, user_message):
        lowered_message = user_message.lower()
        return bool(CODE_REQUEST_PATTERN.search(lowered_message)) or "```" in user_message

    def _load_code_bot(self):
        if self.code_bot is None:
            rich_print(f"[yellow]Loading code model:[/yellow] {self.code_model_name}")
            self.code_bot = PretrainedChatbot(
                self.code_model_name,
                system_prompt=CODE_SYSTEM_PROMPT,
                max_response_tokens=code_max_new_tokens,
            )
        return self.code_bot

    def reply(self, user_message):
        if self._is_code_request(user_message):
            try:
                return self._load_code_bot().reply(user_message)
            except Exception as error:
                rich_print(
                    "[yellow]Code model could not be loaded. Using the general chat model instead.[/yellow]"
                )
                rich_print(f"[dim]{error}[/dim]")
        return self.general_bot.reply(user_message)


def _load_pretrained_chatbot():
    try:
        rich_print(f"[yellow]Loading pretrained chatbot:[/yellow] {pretrained_model_name}")
        return RoutedChatbot(pretrained_model_name, code_model_name)
    except Exception as error:
        rich_print(
            "[yellow]Pretrained chatbot could not be loaded. "
            "Falling back to the local toy model.[/yellow]"
        )
        rich_print(f"[dim]{error}[/dim]")
        return None


def _build_prompt(history, user_message):
    recent_history = history[-conversation_turns:]
    prompt_parts = [f"System: {SYSTEM_PROMPT}"]
    for previous_user, previous_assistant in recent_history:
        prompt_parts.append(f"User: {previous_user}")
        prompt_parts.append(f"Assistant: {previous_assistant}")
    prompt_parts.append(f"User: {user_message}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def _sanitize_text(text):
    return "".join(character if character in stoi else " " for character in text)


def _extract_reply(full_text):
    assistant_marker = "Assistant:"
    reply = full_text.rsplit(assistant_marker, maxsplit=1)[-1]
    for stop_marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
        if stop_marker in reply:
            reply = reply.split(stop_marker, maxsplit=1)[0]
    return _normalize_reply_text(reply)


def generate_reply(model, history, user_message):
    prompt = _sanitize_text(_build_prompt(history, user_message))
    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )[0].tolist()
    generated_text = decode(generated_tokens)
    return _extract_reply(generated_text)


def chat(model=None):
    """Run an interactive terminal chat session."""
    chatbot = None
    if chat_backend == 'pretrained':
        chatbot = _load_pretrained_chatbot()
    if chatbot is None:
        if model is None:
            raise ValueError("A local model is required when the pretrained backend is unavailable.")
        chatbot = LocalChatbot(model)

    console.print(Panel.fit("Chatbot\nType 'exit' to quit.", title="Chat"))
    while True:
        user_message = console.input("[bold cyan]You:[/bold cyan] ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        reply = chatbot.reply(user_message)
        _render_reply(reply)
