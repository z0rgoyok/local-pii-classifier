"""Local binary private-data classifier. No redaction or external-call permission.

PRIVATE: an explicit detector fired. PUBLIC: scan completed without findings.
BLOCK: scan incomplete; this is abstention and never counts as detected private.
The classifier intentionally returns no matched values or normalized offsets.
"""

import html
import ipaddress
import json
import re
import time
import unicodedata
from urllib.parse import urlsplit

POLICY_VERSION = "pii-cascade-v2"
MAX_CHARS = 200_000
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 10_000


def pattern(expression):
    return re.compile(expression, re.IGNORECASE | re.UNICODE)


RULES = {
    "EMAIL": pattern(r"[\w.+-]+\s*@\s*[\w.-]+\.[a-zа-я]{2,}"),
    "OBFUSCATED_EMAIL": pattern(r"\w[\w.+-]*\s*(?:\[at\]|\(at\)|собака)\s*[\w.-]+\s*(?:\[dot\]|\(dot\)|точка)\s*\w+"),
    "PHONE": pattern(r"(?<!\w)\+\d(?:[\d\s().-]){8,24}\d(?!\w)"),
    "PHONE_RU": pattern(r"(?<!\w)[78][\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"),
    "LONG_ID": pattern(r"(?<!\w)\d(?:[ -]?\d){9,19}(?!\w)"),
    "SNILS": pattern(r"\b\d{3}-\d{3}-\d{3}[ -]\d{2}\b"),
    "URL": pattern(r"\b(?:https?|ssh|git|ftp|postgres(?:ql)?|mysql|redis|mongodb(?:\+srv)?)://\S+"),
    "PRIVATE_KEY": pattern(r"-----\s*begin\s+(?:rsa |ec |openssh |dsa )?private key\s*-----"),
    "AUTH": pattern(r"\b(?:bearer|basic)\s+[a-z\d+/_=.\-]{8,}"),
    "JWT": pattern(r"\beyj[a-z\d_-]{5,}\.[a-z\d_-]+\.[a-z\d_-]+"),
    "PROVIDER_KEY": pattern(r"\b(?:gh[pousr]_[a-z\d_]{12,}|github_pat_[a-z\d_]{12,}|sk-[a-z\d_-]{12,}|akia[a-z\d]{16})\b"),
    "SECRET_ASSIGNMENT": pattern(r"(?<!\w)[\"']?(?:[a-z\d]+[_-])*(?:password|passwd|pass|pwd|token|secret|api[_-]?key|client[_-]?secret|authorization|cookie|session[_-]?id|пароль|токен)(?!\w)[\"']?\s*[:=]\s*[\"']?[^\s\"'{}<>]{3,}"),
    "IDENTITY_ASSIGNMENT": pattern(r"[\"']?(?:account[_-]?id|user[_-]?id|username|login|customer[_-]?id|employee[_-]?id|order[_-]?id|device[_-]?id|телефон|паспорт|снилс|инн|лицевой\s+сч[её]т)[\"']?\s*[:=]\s*[\"']?[^\s\"'{}<>]{2,}"),
    "ASSIGNED_PERSON": pattern(r"\b(?:assignee|reporter|author|ответственный|исполнитель|автор|фио|contact)\s*[:=]\s*[\"']?[a-zа-яё][\w.@-]*(?:\s+[a-zа-яё][\w.-]*)?"),
    "INTERNAL_DNS": pattern(r"\b[\w-]+(?:\.[\w-]+)*\.(?:internal|local|corp|lan|intranet)\b"),
    "IPV4": pattern(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    "IPV6": pattern(r"(?<![\w:])[\da-f]*:[\da-f:]+(?![\w:])"),
    "TEST_VALUES": pattern(r"\b(?:test[_-]?data|fixture|тестов\w*\s+(?:данн\w*|пользоват\w*|аккаунт\w*|клиент\w*|заказ\w*))\s*[:=]\s*[^\s<>]{2,}"),
    "ENCODED_SECRET": pattern(r"\b(?:base64|b64|encoded|закодирован\w*)\b.{0,80}(?:[a-z\d+/]{16,}={0,2})"),
    "INJECTION": pattern(r"(?:ignore|disregard|override).{0,60}(?:previous|system|instructions)|(?:игнорир\w*|забудь).{0,50}(?:инструкц\w*|правил\w*)"),
    "SPOKEN_PHONE": pattern(r"(?:семь|восемь)\s+(?:девятьсот|девять\s+сот|ноль)[а-яё\s-]{10,120}"),
}

PERSON = r"(?:сотрудни[кц]\w*|работник\w*|коллег\w*|клиент\w*|пациент\w*|employee|colleague|patient|he|she)"
HEALTH = r"(?:диагноз\w*|диабет\w*|болезн\w*|болеет|лечени\w*|лекарств\w*|больничн\w*|беременн\w*|инвалид\w*|психиатр\w*|депресси\w*|medical\s+diagnosis|therapy|pregnan\w*|disabilit\w*|diagnosed|cancer|hiv)"
HR = r"(?:зарплат\w*|оклад\w*|преми\w*|увольн\w*|дисциплинар\w*|взыскани\w*|испытательн\w*|оценк\w*|salary|dismiss\w*|disciplin\w*|personnel\s+record|performance\s+review)"
SEMANTIC_RULES = {
    "HEALTH_CONTEXT": pattern(PERSON + r".{0,180}" + HEALTH + "|" + HEALTH + r".{0,180}" + PERSON),
    "HR_CONTEXT": pattern(PERSON + r".{0,180}" + HR + "|" + HR + r".{0,180}" + PERSON),
    "SALARY_VALUE": pattern(r"(?:зарплат\w*|оклад\w*|salary|compensation).{0,40}\d[\d .,]*\s*(?:руб\w*|eur|usd|rub|[$€₽])"),
    "PERSONAL_COMPENSATION": pattern(r"(?:зарплат\w*|оклад\w*|salary|compensation).{0,60}\d{3,}(?:[\d .,]*\d)?"),
}

PUBLIC_DOC_HOSTS = {
    "developer.android.com", "developer.apple.com", "kotlinlang.org", "docs.python.org",
    "docs.oracle.com", "www.rfc-editor.org", "rfc-editor.org", "www.w3.org",
    "www.ietf.org", "datatracker.ietf.org", "www.rust-lang.org", "doc.rust-lang.org",
    "docs.example.invalid",
}
UUID_SHAPE = pattern(r"\b[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}\b")


def has_private_url(view):
    for match in RULES["URL"].finditer(view):
        parsed = urlsplit(match.group().rstrip(".,;<>\"')"))
        if (parsed.scheme == "https" and parsed.hostname in PUBLIC_DOC_HOSTS
                and not parsed.query and not parsed.fragment and not parsed.username
                and not UUID_SHAPE.search(parsed.path)):
            continue
        return True
    return False

PRIVATE_KEYS = {
    "assignee", "reporter", "author", "creator", "displayname", "emailaddress",
    "username", "accountid", "userid", "employeeid", "customerid", "orderid",
    "deviceid", "password", "token", "secret", "apikey", "testdata", "fixture",
}
PLACEHOLDERS = {"", "null", "none", "redacted", "masked", "<phone>", "<email>", "<user_id>"}
# DATE/TIME/COMPANY/COUNTRY alone are insufficient evidence under this policy.
NYM_SIGNALS = {
    "GIVEN_NAME", "SURNAME", "USERNAME", "EMAIL", "PHONE", "PASSPORT", "SSN",
    "TAX_ID", "CREDIT_DEBIT_CARD", "CVV", "PIN", "PASSWORD",
    "ACCOUNT_NUMBER", "CUSTOMER_ID", "EMPLOYEE_ID", "DATE_OF_BIRTH",
    "DRIVERS_LICENSE", "MEDICAL_RECORD_NUMBER", "IBAN", "MAC_ADDRESS",
    "STREET_ADDRESS", "STREET_NAME", "SECONDARY_ADDRESS",
}
ADDRESS_CONTEXT = pattern(r"\b(?:адрес\w*|прожив\w*|место\s+рожден\w*|address|residen\w*|street|born)\b")
A1_LOCATION_ONLY = {"COUNTRY", "CITY", "REGION", "DISTRICT", "STREET", "HOUSE", "APARTMENT", "POSTCODE"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    for _ in range(3):
        text = "".join(c for c in text if unicodedata.category(c) != "Cf")
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    # Strip actual markup while preserving non-HTML placeholders.
    text = re.sub(r"</?(?:p|div|span|a|pre|code|br|li|ul|ol|b|strong|em|table|tr|td)\b[^>]*>", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", "".join(c for c in text if unicodedata.category(c) != "Cf"))


def json_strings(text):
    """Decode escaped JSON, including nested history strings; bound all recursion."""
    findings, leaves = set(), []
    nodes = 0

    def walk(value, depth):
        nonlocal nodes
        nodes += 1
        if depth > MAX_JSON_DEPTH or nodes > MAX_JSON_NODES:
            raise ValueError("json_limit")
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = re.sub(r"[_-]", "", str(key)).lower()
                if normalized_key in PRIVATE_KEYS and child is not None and child != "" and child != []:
                    findings.add("STRUCTURED_PRIVATE_FIELD")
                walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth + 1)
        elif isinstance(value, str):
            normalized = normalize(value)
            if normalized.lstrip().startswith(("{", "[")):
                try:
                    nested = json.loads(normalized)
                except (ValueError, RecursionError):
                    leaves.append(normalized)
                else:
                    walk(nested, depth + 1)
            else:
                leaves.append(normalized)

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            return set(), [text]
        walk(value, 0)
        return findings, leaves
    return findings, [text]


class Classifier:
    def __init__(self, a1=None, nym=None, rules_only=False):
        self.a1, self.nym = a1, nym
        self.rules_only = rules_only

    def classify(self, text, source="text"):
        started = time.perf_counter()

        def result(verdict, reasons):
            return {"verdict": verdict, "contains_private": True if verdict == "PRIVATE" else False if verdict == "PUBLIC" else None,
                    "reasons": sorted(reasons), "policy": POLICY_VERSION,
                    "elapsed_ms": (time.perf_counter() - started) * 1000}

        if not isinstance(text, str) or len(text) > MAX_CHARS:
            return result("BLOCK", {"INPUT_LIMIT"})
        if source in {"attachment_binary", "image", "audio", "video", "pdf"}:
            return result("BLOCK", {"UNSUPPORTED_CONTENT"})
        try:
            normalized = normalize(text)
            findings, leaves = json_strings(normalized)
            # Preserve the raw view too: malformed structured input still scans.
            views = [normalized] + leaves
            for view in views:
                # Deleted config values remain private in Git history/diffs.
                # Numeric tuning values are excluded from this conservative rule.
                if source == "git_diff" and re.search(
                    r"(?m)(?<![\w-])-\s*[\w.-]+\s*=\s*[a-zA-Z][\w.-]{5,}", view
                ):
                    findings.add("REMOVED_CONFIG_VALUE")
                for label, regex in {**RULES, **SEMANTIC_RULES}.items():
                    if label == "URL":
                        matched = has_private_url(view)
                    elif label == "LONG_ID":
                        matched = regex.search(UUID_SHAPE.sub("UUID", view))
                    elif label in {"IPV4", "IPV6"}:
                        matched = False
                        for match in regex.finditer(view):
                            try:
                                ipaddress.ip_address(match.group())
                            except ValueError:
                                continue
                            matched = True
                            break
                    else:
                        matched = regex.search(view)
                    if matched:
                        findings.add(label)
            if findings:
                return result("PRIVATE", findings)
            if self.rules_only:
                return result("PUBLIC", set())
            if self.a1 is None or self.nym is None:
                return result("BLOCK", {"MODEL_UNAVAILABLE"})
            for leaf in leaves:
                a1_labels = self.a1.predict(leaf)
                if not ADDRESS_CONTEXT.search(leaf):
                    a1_labels -= A1_LOCATION_ONLY
                if a1_labels:
                    return result("PRIVATE", {"A1_" + label for label in a1_labels})
                nym_labels = self.nym.predict(leaf, min_entity_mass=0.80) & NYM_SIGNALS
                if nym_labels:
                    return result("PRIVATE", {"NYM_" + label for label in nym_labels})
            return result("PUBLIC", set())
        except Exception:
            # Exception text may contain input or runtime paths; never print it.
            return result("BLOCK", {"SCAN_ERROR"})
