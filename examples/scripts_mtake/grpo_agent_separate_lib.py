import re
import signal
import sqlite3
from contextlib import contextmanager


def query_reward(completions, answer, **kwargs):
    """
    Reward query strategy:
    - Penalize more than 2 queries
    - Penalize generic queries (LIMIT 1 / PRAGMA)
    - Reward usage of WHERE
    - Reward evidence supporting the final answer
    """
    rewards = []

    for completion, ans in zip(completions, answer, strict=False):
        reward = 0.0
        sql_queries = []
        tool_results = []

        # collect all SQL queries and tool results
        for turn in completion:
            if turn.get("tool_calls"):
                for call in turn["tool_calls"]:
                    sql = call["function"]["arguments"].get("sql_command", "").lower()
                    sql_queries.append(sql)
            if turn.get("role") == "tool" and turn.get("content"):
                tool_results.append(turn["content"])

        # --- penalize too many queries ---
        if len(sql_queries) > 3:
            reward -= 1.5

        # --- check query quality ---
        where_count = 0
        for q in sql_queries:
            if "limit 1" in q:
                reward -= 1.0
            if " where " not in q:
                reward -= 0.5
            else:
                where_count += 1
        reward += min(where_count, 3) * 0.4  # small bonus for WHERE usage

        # --- evidence check: do queries support the answer? ---
        combined_results = []
        error_detected = False

        for res in tool_results:
            if isinstance(res, dict) and "error" in res:
                error_detected = True
            elif isinstance(res, list):
                combined_results.extend(res)

        # if error detected, penalize heavily
        if error_detected:
            reward -= 2.0
        elif len(sql_queries) == 0:
            reward -= 1.5
        else:
            has_hits = len(combined_results) > 0
            correct_answer = ans.lower()
            if (has_hits and correct_answer == "yes") or (not has_hits and correct_answer == "no"):
                reward += 2.0
            else:
                reward -= 1.5

        rewards.append(reward)

    return rewards


def correctness_reward(completions, answer, **kwargs):
    """
    Reward Yes/No correctness.
    Model must provide final answer enclosed in stars — *yes* or *no*.
    Does not reward informal yes/no buried in text.
    """
    rewards = []
    for completion, ans in zip(completions, answer, strict=False):
        # @@@ahoaho XXX granite-4.2 could return completion[-1]["content"] as list, which will cause an error.
        # print(f"XXX completion[-1] = XXX{completion[-1]}XXX")
        raw = completion[-1]["content"].lower()

        # detect form *yes* or *no*
        match = re.search(r"\*(yes|no)\*", raw)
        guess = match.group(1) if match else None

        reward = 0.0

        if guess is None:
            reward -= 0.5  # invalid format
        elif guess == ans.lower():
            reward += 0.6  # correct under required format
        else:
            reward -= 1.0  # wrong answer

        rewards.append(reward)

    return rewards


def structure_reward(completions, **kwargs):
    """
    Reward proper assistant structure.
    Encourages a logical sequence: tool call + response + optional extra content.
    """
    rewards = []

    for completion in completions:
        has_call = False
        has_response = False
        has_other = False

        for turn in completion:
            role = turn.get("role")
            if role == "assistant" and turn.get("tool_calls"):
                has_call = True
            elif role == "tool":
                has_response = True
            else:
                content = turn.get("content")
                if content and content.strip() not in ["", "<think>"]:
                    has_other = True

        # Reward sequences
        if has_call and has_response:
            if has_other:
                reward = 0.1
            else:
                reward = 0.05  # still positive even without extra text
        elif has_call and not has_response:
            reward = -0.15
        else:
            reward = 0.0  # neutral if no call

        rewards.append(reward)

    return rewards


# ------------------------
# Database tool function
# ------------------------
class TimeoutError(Exception):
    """Raised when a function call times out."""

    pass


@contextmanager
def timeout(seconds):
    """Context manager that raises TimeoutError if execution exceeds time limit."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def query_biogrid(sql_command: str) -> list[tuple]:
    """
    Execute a read-only SQL command on the BioGRID database.

    BioGRID is a curated biological database that compiles protein, genetic, and chemical interactions from multiple organisms. It provides researchers with experimentally verified interaction data to support studies in systems biology and functional genomics.

    Args:
        sql_command: The SQL command to execute.

    Returns:
        A list of tuples containing the query results.
    """
    with timeout(5):
        conn = sqlite3.connect("file:biogrid.db?mode=ro", uri=True)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_command)
            results = cursor.fetchall()
        finally:
            conn.close()
    return results
